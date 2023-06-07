# import boto3
import json
import numpy as np
import requests


def serverless_inference(text):
    url = 'https://p3p9wvyk5h.execute-api.us-west-2.amazonaws.com/stage-1/api-endpoint-inference'

    payload = {
        "body": text
    }

    headers = {
        "Content-Type": "application/json"
    }

    # print (json.loads(response["Body"].read()))

    payload_json = json.dumps(payload)

    # Set the appropriate headers for JSON content
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send the POST request with the JSON payload
        response = requests.post(url, data=payload_json, headers=headers)
        # print(response.text)
    except Exception as e:
        print("An unexpected error occoured")

    response_body = response.text.encode("utf-8")
    data_list = json.loads(response_body)
    nested_list = json.loads(data_list)

    numpy_array = np.array(nested_list)
    numpy_array = numpy_array[0][0]

    numpy_array = np.reshape(numpy_array, (1, 768))
    print(numpy_array.shape)
    return (numpy_array)

