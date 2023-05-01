from flask import Flask, render_template, request, url_for
from piidetection import OpenAI
from piidetection import Piidetector
import json
import requests

app = Flask(__name__)

model = OpenAI(api_key='OpenAI key ')
nlp_prompter = Piidetector(model)
ocrKey = 'OCR SPACE KEY'

@app.route('/', methods=["GET", "POST"])
def detectPII():
    global result
    result = ''
    text_input = ''
    active_tab = 'text'
    if request.method == "POST":
        examples = []
        if len(request.form) > 0:

            text_input = request.form['pii-text']
            json_file = request.files['json-file']
            if json_file and json_file.filename.endswith('.json'):
                data = json.load(json_file)
                for sample in data[:len(data)]:
                    examples.append((sample['text'],sample['labels']))
        else:
            active_tab = 'file'
            file_data = request.files['pii-file']
            text_input = extractText(file_data)
        output = nlp_prompter.fit('binary_classification.jinja',
                 label_0="pii",
                 label_1="non-pii",
                 examples=examples,
                 text_input=text_input,
                 model_name="text-davinci-003")
        result = eval(output['text'].strip())[0]['C'].lower()
    else:
        result = ''
    return render_template('index.html', result=result, active_tab=active_tab)

def extractText(file_data):
    payload = {'isOverlayRequired': False,
               'apikey': ocrKey,
               'language': 'eng',
               }
    r = requests.post('https://api.ocr.space/parse/image',
                files={file_data.filename: file_data.read()},
                data=payload,
            )
    list = json.loads(r.content.decode())['ParsedResults']
    text = ''
    for sample in list:
        text = text+sample['ParsedText']
        print(sample['ParsedText'])
        print('\n')
    return text