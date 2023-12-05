from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import torch

app = Flask(__name__)

# Global variables for model components
model = image_processor = text_processor_infer = None

def load_model():
    # Placeholder for model loading logic
    # Replace this with actual model loading code
    global model, image_processor, text_processor_infer
    model = None  # Initialize your model here
    image_processor = None  # Initialize your image processor here
    text_processor_infer = None  # Initialize your text processor here

@app.route('/chat', methods=['POST'])
def chat_api():
    content = request.json
    input_text = content.get('input_text', '')
    temperature = content.get('temperature', 0.8)
    top_p = content.get('top_p', 0.4)
    top_k = content.get('top_k', 10)
    image_data = content.get('image', None)

    if image_data:
        pil_img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    else:
        pil_img = None

    try:
        with torch.no_grad():
            response, _, _ = chat(
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=input_text,
                history=[],
                image=pil_img,
                max_length=2048,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                no_prompt=False
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": response})

if __name__ == '__main__':
    load_model()  # Load the model when starting the app
    app.run(debug=True, port=5000)
