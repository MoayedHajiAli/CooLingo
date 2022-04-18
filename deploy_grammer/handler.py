import json
from grammar_correction import GrammarCorrector
import logging

model = GrammarCorrector()

sample_text = 'I am try this code'
print("Sample text:", sample_text)
print("Corrected text:", model.correct_grammar(sample_text))

def predict(event, context):
    print("Event", event)
    print("context", context)
    try:
        body = json.load(event['body'])
        context.log(body)
        correct_sentence = model.correct_grammar(body['input'])
        context.log(correct_sentence)
        logging.info(f'Sentence {body['input']}')
        logging.info(f'Prediction {correct_sentence}')

        

        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True},
            "body": json.dumps({"correct_sentence":correct_sentence})
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
