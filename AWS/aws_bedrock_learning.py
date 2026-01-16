import boto3
import json

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
model_id = "anthropic.claude-haiku-4-5-20251001-v1:0"

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "temprature": 0.5,
    "max_tokens": 500,
    "messages": [
        {
            "role": "user",
            "content": "Hello! Can you tell me a short joke?"
        }
    ]
})

response = client.invoke_model(
    modelId=model_id,
    body=body,
    accept="application/json",
    contentType="application/json"
)

print(response)