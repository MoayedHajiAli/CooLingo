import json
from blenderbot import Blenderbot
import logging

model = Blenderbot(device='cpu')

sample_text = 'Hello there!'
sample_starter = 'hello'
print("Sample text:", sample_text)
print("Sample starter:", sample_starter)
print("Sample response:", model.get_response(sample_text, sample_starter))

def predict(event, context):
    print("Event", event)
    try:
        print("event", event['body'])
        body = json.loads(event['body'])
        context.log(body)

        message = body.get('input', '')
        reply_start = body.get('prepend', None)

        reply, past = model.get_response(message, reply_start=reply_start)
        context.log(reply)
        print("reply:", reply)

        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True},
            "body": json.dumps({"reply":reply})
        }

    except Exception as e:
        logging.error(e)
        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True},
            "body": json.dumps({"error":repr(e)})
        }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
