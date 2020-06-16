from flask import Flask, request, jsonify
import time
import kg_extraction as askbert
#from nltk import sent_tokenize
app = Flask(__name__)
askbert_args = {'input_text' : '', 'length':10, 'batch_size': 1, 'temperature':1, 'model_name':'117M',
            'seed':0, 'nsamples':10, 'cutoffs':"6 7 5", 'write_sfdp':False, 'random':False}
#world = extraction.World(askbert_args)
#import timeit
import kg_extraction

@app.route('/', methods=['POST'])
def result():
    count = 1
    if request.method == 'POST':
        data = request.form
        entities = kg_extraction.generate(data['state'], float(data['threshold']), bool(data['attribute']))
        ov = {'entities': entities}
        return jsonify(ov), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0')