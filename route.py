# -*- coding: utf-8 -*-
from transformers import GPT2Tokenizer, GPTNeoModel
import faiss
import logging
from flask import Flask, g, _app_ctx_stack
from flask_cors import CORS
from flask import request
from flask import jsonify
import argparse
import json
import utils as utils
import serverless as serverless
import os
import numpy as np

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

app = Flask(__name__)
app.config.from_object('settings')
CORS(app)

@app.before_first_request
def initialize():
    global SRC_PATH, MODEL, INDEX, ID2LINE, TOKENIZER
    
    try:
        path = app.config["SRC_PATH"]
    except:
        try: path = SRC_PATH
        except: 
            app.logger.error("SRC_PATH Not found")
            raise Exception("SRC_PATH Not found")

    app.logger.warning("Loading Core files from {}".format(path))
    app.logger.warning("Loading gpt-neo-125M model and tokenizer and faiss index")
    # model_id = "EleutherAI/gpt-neo-125M"
    # TOKENIZER = GPT2Tokenizer.from_pretrained(model_id)
    # TOKENIZER.pad_token = TOKENIZER.eos_token
    # MODEL = GPTNeoModel.from_pretrained(model_id)
    app.logger.warning("Loading mapfile from {}".format(path))
    ID2LINE = utils.load_id2line_map(path)

    app.logger.warning("Loading index from {}".format(path))
    INDEX = faiss.read_index(path+"/BigBird-Patent-Index-ABV.index")
    

    app.logger.warning("Loaded all files in memory")


@app.route("/")
def index():
    return "API working.", 200
    

@app.route("/nexus/r1/index/ClaimsTextSearch4/search", methods=['POST'])
def search_index():

    print("Seaching")
    # if request.headers.get("X-Amzn-Apigateway-Api-Id") != app.config["APIGATEWAY_ID"]:
    #     app.logger.error("Forbidden access")
    #     return "Forbidden", 403
    req = request.get_json()
    req = req["request"]
    app.logger.debug("Incoming Request: {}, {}".format(request.method, req))
    if request.method != 'POST':
        app.logger.error("Invalid request method received")
        return "Wrong Request", 400

    # TODO Add limit and validation
    if "maxNumItems" in req:
        max_result = int(req["maxNumItems"])
    else:
        max_result = 1000

    try:
        # vector = utils.vectorize_text(MODEL,TOKENIZER,req['conceptFeaturesString'])
        vector = utils.vectorize_text_sagemaker_endpoint(req['conceptFeaturesString'])
        # vector = serverless.serverless_inference(req[''])

    except KeyError:
        app.logger.exception("BadRequest!!! check your query string.")
        return "BadRequest!!! check your query string.", 400

    app.logger.info("RequestParsed: text {}, max_result {}".format(
        req["conceptFeaturesString"][:15], max_result))
    
    dists, ids = INDEX.search(
                np.array(vector).astype("float32"), k=max_result)
    results = list(ids[0])
    score = list(dists[0])

    print(results)
    print(score)

    # app.logger.debug("QueryD2V: Completed, starting Index threads. Result dimensions are: {}".format(len(vector)))
    # app.logger.debug("QueryD2V: Results are: {}".format(vector))
    # app.logger.debug("Results: Results are: {}".format(results))

    desired_results = results
    total_results = len(results)
    app.logger.debug("Results are {}".format(total_results))

    output = {
      "totalFoundItems": str(total_results),
      "foundItems": [{
        # "app_num": ID2LINE[str(i+1)].get("app_num", "").strip(),
        # "pub_num": ID2LINE[str(i+1)].get("pub_num", "").strip(),
        # "grant_num": ID2LINE[str(i+1)].get("grant_num", "").strip(),
        "ucid": ID2LINE[str(i+1)].get("ucid", "").strip(),
        "id": ID2LINE[str(i + 1)].get("id", "").strip(),

          # "application_number": ID2LINE[str(i+1)].get("application_number", "").strip(),
        # "publication_date": ID2LINE[str(i+1)].get("publication_date", "").strip(),
        # "filing_date": ID2LINE[str(i+1)].get("filing_date", "").strip(),
        # "grant_date": ID2LINE[str(i+1)].get("grant_date", "").strip(),
        # "priority_date": ID2LINE[str(i+1)].get("priority_date", "").strip(),
        # "assignee": ID2LINE[str(i+1)].get("assignee", "").strip(),
        # "priority_number": ID2LINE[str(i+1)].get("priority_number", "").strip(),
        "conceptScore": str(s)
      } for i, s in zip(desired_results,score) if (i != 0)]
    }

    print(output)

    app.logger.info("Response: {}".format(output["foundItems"]))
    return json.dumps(output), 200, {'Content-Type': 'application/json; charset=utf-8'}


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

  

@app.route("/vectors", methods=['POST'])
def get_vectors():
    if request.headers.get("X-Amzn-Apigateway-Api-Id") != app.config["APIGATEWAY_ID"]:
        app.logger.error("Forbidden access")
        return "Forbidden", 403
    req = request.get_json()
    req = req["request"]
    app.logger.debug("Incoming Request: {}, {}".format(request.method, req))
    if request.method != 'POST':
        app.logger.error("Invalid request method received")
        return "Wrong Request", 400
    try:
        query = req["query"]
    except KeyError:
        app.logger.exception("Bad Request! Check your query string!")
        return "Bad Request", 400
    vector= utils.transform_query(query, MODEL)
    vec_json = {
        "query": query,
        "vector representation": vector.tolist()
    }
    response = jsonify(vec_json)
    response.status_code = 200
    return response

if __name__ == "__main__":
    global SRC_PATH
    # Parse arguments
    parser = argparse.ArgumentParser(description='Server lsi indexes via api.')
    parser.add_argument('src_path', metavar='Source path to the file.', type=str,
        help='Source/Input data path such as /tmp/file-name')
    parser.add_argument('--host', type=str, default='0.0.0.0',
        help='Adjust the host of the server.')
    parser.add_argument('--port', type=int, default=80,
        help='Adjust the port of the server.')
    parser.add_argument('--debug', type=bool, default=False,
        help='Enable app debugging.')

    # args = parser.parse_args()
    # File path validations
    # SRC_PATH = os.path.abspath(args.src_path)
    # app.run(host=args.host, port=args.port, debug=args.debug)
    app.run(port=3000)