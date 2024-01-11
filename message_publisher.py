from flask import Flask, request, jsonify
import pika
import json
import argparse
import uuid
import time

app = Flask(__name__)

def publish_message(image_path, query, history):
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='chat_queue', durable=False)
    
    # Generate a unique ID for the message
    message_id = str(uuid.uuid4())

    message = {
        'id': message_id,
        'image_path': image_path,
        'query': query,
        'history': history
    }

    channel.basic_publish(
        exchange='',
        routing_key='chat_queue',
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )

    print(f" [x] Sent {message}")
    connection.close()
    return message_id

def wait_for_reply(message_id):
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='reply_queue', durable=False)

    start_time = time.time()
    timeout = 30  # seconds

    while time.time() - start_time < timeout:
        method_frame, header_frame, body = channel.basic_get(queue='reply_queue')
        if method_frame:
            channel.basic_ack(method_frame.delivery_tag)
            body = json.loads(body)
            if body.get('request_id') == message_id:
                return body.get('response')
        time.sleep(1)

    connection.close()
    return None

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    image_path = data.get('image_path')
    query = data.get('query')
    history = data.get('history')

    if not image_path or not query:
        return jsonify({'error': 'Missing image_path or query'}), 400

    message_id = publish_message(image_path, query, history)
    reply = wait_for_reply(message_id)

    if reply:
        return jsonify({'reply': reply})
    else:
        return jsonify({'error': 'No reply received within timeout period'}), 504

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

