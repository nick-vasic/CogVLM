# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel

import pika
import json
from PIL import Image
import requests
from io import BytesIO

from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel

def get_next_message():
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='chat_queue', durable=False)

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

    channel.queue_declare(queue='reply_queue', durable=False)

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
    parser.add_argument("--chinese", action='store_true', help='Chinese interface')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    print("[Language processor version]:", language_processor_version)
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    
    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()


    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    if rank == 0:
        print('*********** LISTENING FOR REQUESTS ***********')
        
    with torch.no_grad():
        while True:
            time.sleep(0.5)
            history = None
            cache_image = None
            
            if rank == 0:
                next_message = get_next_message()
                if next_message is None:
                    continue
                image_path = next_message.get('image_path', '')
                print('Message received: ' + next_message['id'])
                if not is_valid_image(image_path):
                    post_reply('Not a valid image: ' + image_path, [], next_message['id'])
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
                
            try:
                response, history, cache_image = chat(
                        image_path,
                        model,
                        text_processor_infer,
                        image_processor,
                        query,
                        history=history,
                        cross_img_processor=cross_image_processor,
                        image=cache_image,
                        max_length=args.max_length,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        args=args
                        )
            except Exception as e:
                print(e)
                break
            if rank == 0:
                post_reply(response, history, next_message['id'])

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
