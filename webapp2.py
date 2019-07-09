import tornado.ioloop
import tornado.web
from collections import namedtuple
from inference_api_tensor_out import predict_wrapper
import json
import os
import numpy as np
import array
# from logger_util import *

curDir = os.path.dirname(os.path.realpath(__file__))


def jsonToObj(jsonObj):
    return json.loads(json.dumps(jsonObj), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

#
# argsJson = {
#     'beam_size': 4,
#     'bert_config': os.path.join(curDir, "..", "model", "bert_config.json"),
#     'do_lower_case': True,
#     'init_model': os.path.join(curDir, "..", "model", "pytorch.model-9.bin"),
#     'max_seq_length': 256,
#     'tag': 'WikiSQL_Eval',
#     'vocab': os.path.join(curDir, "..", "model", "vocab.txt"),
# }
# args = jsonToObj(argsJson)
# modelWrapper = ModelWrapper(args)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("SQL Max 6/21/2019")


# http://localhost:8888/query?question=What position does the player who played for butler cc (ks) play&header=["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]&types=["text", "text", "text", "text", "text", "text"]
# question: What position does the player who played for butler cc (ks) play?
# header: ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]
# types: ["text", "text", "text", "text", "text", "text"]

class QueryHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        inputQuery = {
            'question': self.get_argument("question"),
            'header': json.loads(self.get_argument("header", default="[]")),
            'types': json.loads(self.get_argument("types", default="[]")),
            'data_ix': json.loads(self.get_argument("data_ix", default="[]")),
        }

        select, wcn, wcond = predict_wrapper(inputQuery)

        self.write({
            "select": select,
            "wcn": wcn,
            "where": wcond
        })

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/query", QueryHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    port = 8888
    app.listen(port)
    print("start port:" + str(port))
    tornado.ioloop.IOLoop.current().start()