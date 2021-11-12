import boto3

client = boto3.client('sagemaker-runtime')

custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "pytorch-inference-2021-11-12-01-18-29-802"                                       # Your endpoint name.
content_type = "text/plain"                                        # The MIME type of the input data in the request body.
accept = "text/plain"                                              # The desired MIME type of the inference in the response.
payload = "test input"                                             # Payload for inference.
response = client.invoke_endpoint(
    EndpointName=endpoint_name, 
    CustomAttributes=custom_attributes, 
    ContentType=content_type,
    Accept=accept,
    Body=payload
    )

print(response)          

print(response['Body'].read())