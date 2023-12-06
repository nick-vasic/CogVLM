from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64
import torch
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size
from utils.parser import parse_response
from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

app = Flask(__name__)

# Global variables for model components
model = image_processor = text_processor_infer = None

def load_model(args, rank, world_size): 
    global model, image_processor, text_processor_infer 
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            fp16=args.fp16,
            bf16=args.bf16,
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device=f'cuda'
        ),
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    return model, image_processor, text_processor_infer


@app.route('/chat', methods=['POST'])
def chat_api():
    content = request.json
    input_text = content.get('input_text', '')
    temperature = content.get('temperature', 0.8)
    top_p = content.get('top_p', 0.4)
    top_k = content.get('top_k', 10)
    image_data = content.get('image', None)
    image_path = content.get('image_path', None)

    if image_data:
        pil_img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    else:
        pil_img = None

    try:  
        with torch.no_grad():
            response, _, _ = chat(
                image_path=image_path,
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=input_text,
                history=None,
                image=pil_img,
                max_length=2048,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                invalid_slices=text_processor_infer.invalid_slices,
                no_prompt=False
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": response})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    load_model(args, rank, world_size)  # Load the model when starting the app
    # Start the Flask app only if the rank is 0
    if rank == 0:
        app.run(debug=False, port=5000)  # Set debug to False in production
    else:
        # For non-zero ranks, you can perform other tasks or simply do nothing
        pass