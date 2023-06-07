import os
import torch
import json
import numpy as np
import requests

def load_id2line_map(path):
    """
    Load the main ID of the document to line number
    @return Dictionary
    """

    map_result = {}
    for filename in os.listdir(path):
      if filename.endswith("3p_map_file.csv"):
        with open(os.path.join(path, filename), 'r', encoding="utf-8") as fin:
           lines = fin.readlines()      
        for line in lines:
          #   x = line.split("\t")
          #   print(x)
          #
          #   ucid = x[4].strip('""\n\r')
          #   if len(ucid) > 14:
          #       ucid = ucid[:7]+'0'+ucid[7:]
          #   map_result[str(x[0])] = {
          #   "ucid": ucid,
          #   "application_number": x[5].strip('""\n\r'),
          #   "priority_date": x[9].strip('""\n\r'),
          #   "assignee": x[10].strip('""\n\r'),
          #   "priority_number": x[11].strip('""\n\r')
          # }
          x = line.split(",")

          ucid = x[2].strip('""\n\r')
          if len(ucid) > 14:
              ucid = ucid[:7] + '0' + ucid[7:]

          map_result[str(x[0])] = {
              "id": str(x[1].strip('""\n\r')),
              "ucid": str(ucid),
          }
    return map_result



def vectorize_text(model,tokenizer,text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    outputs = model(**inputs)
    vector = torch.mean(outputs.last_hidden_state, dim=1)
    vector = vector.detach().cpu().numpy()
    return vector

def vectorize_text_sagemaker_endpoint(text):

    url = 'https://p3p9wvyk5h.execute-api.us-west-2.amazonaws.com/stage-1/api-endpoint-inference'
    # url = 'https://wtjvtdttdl.execute-api.us-west-2.amazonaws.com/test/endpoint-inference'
    payload ={'inputs': text,'ucid':"US-9996715-B2	", 'model_name':'bigbird-pegasus-large-bigpatent', 'index':12}
    headers = {"Content-Type": "application/json"}

    payload_json = json.dumps(payload)


    try:
        response = requests.post(url, data=payload_json, headers=headers)
    except:
        pass

    response_body = response.text.encode("utf-8")
    data = json.loads(response_body)
    # data = json.loads(data)
    print(data)


    # vector = np.array(data)
    # vector = vector[0][0]

    vector = data['vectors']

    search_query_vectors = np.reshape(vector, (1, 1024))
    print(search_query_vectors.shape)

    return (search_query_vectors)