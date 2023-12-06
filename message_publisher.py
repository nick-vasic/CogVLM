import pika
import json
import argparse

def publish_message(image_path, input_text):
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='chat_queue', durable=True)

    message = {
        'image_path': image_path,
        'input_text': input_text
    }

    channel.basic_publish(
        exchange='',
        routing_key='chat_queue',
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )

    print(f" [x] Sent {message}")
    connection.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Publish a message for image processing.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('input_text', type=str, help='Input text for the chat')

    args = parser.parse_args()

    publish_message(args.image_path, args.input_text)
