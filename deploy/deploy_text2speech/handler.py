import json
from glowTts import GlowTts
import logging

model = GlowTts(device='cpu')
sample_wav = model.generate_voice("Hello there!")[-1]
print("Successfully generated sample voice")

def predict(event, context):
    try:
        print("Logging: event body:", event['body'])
        body = json.loads(event['body'])
        context.log(body)
        context.log("Synthesising voice")
        synthesis_wav = model.generate_voice(body['input'], 
                                            body.get('legnth_scale', 1), 
                                            body.get('noise_scale', 0.33),
                                            use_cuda=False,
                                            enable_figures=False)[-1]

        context.log("Successfully generated voice")
        print("Successfully generated voice")

        

        response = {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True},
            "body": json.dumps({"synthesised_wav":synthesis_wav.tolist(), 
                                "sampling_rate": model.TTS_CONFIG.audio['sample_rate']})
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
