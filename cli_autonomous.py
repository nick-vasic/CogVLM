# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
import pika
import json
from PIL import Image
import requests
from io import BytesIO

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

def get_next_message():
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='chat_queue', durable=True)

    method_frame, header_frame, body = channel.basic_get(queue='chat_queue')
    if method_frame:
        channel.basic_ack(method_frame.delivery_tag)
        return json.loads(body)
    else:
        return None
    
def post_reply(response, history, request_message_id):
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='reply_queue', durable=True)

    message = {
        'response': response,
        'history': history,
        'request_id': request_message_id
    }
    channel.basic_publish(exchange='', routing_key='reply_queue', body=json.dumps(message))

    connection.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat-v1.1", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device='cuda',
            **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    if rank == 0:
        print('Welcome to CogVLM-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
        
    with torch.no_grad():
        while True:
            time.sleep(2)
            history = None
            cache_image = None
            
            if rank == 0:
                next_message = get_next_message()
                if next_message is None:
                    continue
                image_path = next_message.get('image_path', '')
                if not is_valid_image(image_path):
                    post_reply('Not a valid image: ' + image_path, next_message['id'])
                    continue
            else:
                image_path = None

            if world_size > 1:
                image_path_broadcast_list = [image_path]
                torch.distributed.broadcast_object_list(image_path_broadcast_list, 0)
                image_path = image_path_broadcast_list[0]

            assert image_path is not None

            if rank == 0:
                query = next_message.get('query', '')
            else:
                query = None
                
            if world_size > 1:
                query_broadcast_list = [query]
                torch.distributed.broadcast_object_list(query_broadcast_list, 0)
                query = query_broadcast_list[0]
            
            assert query is not None
                
            if rank == 0:
                history = next_message.get('history', [])
            else:
                history = []
            
            if world_size > 1:
                history_broadcast_list = [json.dumps(history)]
                torch.distributed.broadcast_object_list(history_broadcast_list, 0)
                history = json.loads(history_broadcast_list[0])
  
            print("HISTORY" + json.dumps(history))
            try:
                response, history, cache_image = chat(
                    image_path,
                    model,
                    text_processor_infer,
                    image_processor,
                    query,
                    history=history,
                    image=cache_image,
                    max_length=args.max_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    no_prompt=args.no_prompt
                )
            except Exception as e:
                print(e)
                break
            if rank == 0:
                post_reply(response, history, next_message['id'])
                if tokenizer.signal_type == "grounding":
                    print("Grounding result is saved at ./output.png")

def is_valid_image(image_path):
    try:
        # Check if image_path is a URL
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            response.raise_for_status()  # Raise an error for bad status codes
            with Image.open(BytesIO(response.content)) as img:
                img.verify()  # Verify that it's an image
        else:
            # Local file path
            with Image.open(image_path) as img:
                img.verify()  # Verify that it's an image
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main()
