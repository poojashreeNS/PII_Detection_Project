from flask import Flask, render_template, request, url_for
from piidetection import OpenAI
from piidetection import Piidetector
import json

app = Flask(__name__)

model = OpenAI(api_key='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')
nlp_prompter = Piidetector(model)

@app.route('/', methods=["GET", "POST"])
def detectPII():
    global result
    result = ''
    if request.method == "POST":
        examples = []
        text_input = request.form['pii-text']
        json_file = request.files['json-file']
        if json_file and json_file.filename.endswith('.json'):
            data = json.load(json_file)
            for sample in data[:len(data)]:
                examples.append((sample['text'],sample['labels']))
        output = nlp_prompter.fit('binary_classification.jinja',
                 label_0="pii",
                 label_1="non-pii",
                 examples=examples,
                 text_input=text_input,
                 model_name="text-davinci-003")
        result = eval(output['text'].strip())[0]['C'].lower()
    else:
        result = ''
    return render_template('index.html', result=result)