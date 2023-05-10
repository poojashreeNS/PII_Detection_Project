from flask import Flask, render_template, request, url_for
from piidetection import OpenAI
from piidetection import Piidetector
import json
import requests
import numpy as np
import faiss
import openai
import os
import ast

app = Flask(__name__)

openai_key = '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'
ocrKey = 'K85726806988957'
model = OpenAI(api_key = openai_key)
nlp_prompter = Piidetector(model)
openai.api_key = openai_key
# Define the path to the index file
index_path = "vector.index"
custom_pii_path = "custom_pii.text"


@app.route('/', methods=["GET", "POST"])
def detectPII():
    global result
    result = ''
    text_input = ''
    active_tab = 'generic'
    test_input = ''
    matched=''
    score = 0
    custom_texts = []
    custom_error=False
    if request.method == "POST":
        if len(request.files) < 2: #check if tab1
            active_tab = 'generic'
            if(request.form['generic-text'] != ''): # if input is text
                text_input = request.form['generic-text']
            else: # if input is file
                pii_files = request.files.getlist('generic-file')
                text_input = extractText(pii_files)
            output = nlp_prompter.fit('binary_classification.jinja',
                 label_0="pii",
                 label_1="non-pii",
                 text_input=text_input,
                 model_name="text-davinci-003")
            result = eval(output['text'].strip())[0]['C'].lower()
            return render_template('index.html', custom_error=custom_error, result=result, test_input=test_input, matched=matched, score=score, active_tab=active_tab)
        else: #check if tab2
            active_tab = 'custom'

            if request.files.getlist('custom-pii-file')[0].filename != '':
                custom_pii_files = request.files.getlist('custom-pii-file')
                custom_input = extractText(custom_pii_files)
                if os.path.isfile(custom_pii_path):
                    with open(custom_pii_path, "r+") as f:
                        tmp = f.read()
                        custom_texts = ast.literal_eval(tmp)
                        for text in custom_input:
                            if text not in custom_texts:
                                custom_texts.append(text)
                        # custom_texts.extend(custom_input)
                    with open(custom_pii_path, "w") as f:
                        f.write(str(custom_texts))
                else:
                    with open(custom_pii_path, "w") as f:
                        custom_texts.extend(custom_input)
                        f.write(str(custom_texts))

                if os.path.isfile(index_path):
                    # Load the index from the file
                    index = faiss.read_index(index_path)

                # Generate embeddings for the custom inputs
                embeddings = np.array([get_embedding(text) for text in custom_texts])

                # Check if the index file exists
                if os.path.isfile(index_path):
                    # Load the index from the file
                    index = faiss.read_index(index_path)
                else:
                    # Create a new index
                    index = faiss.IndexFlatIP(embeddings.shape[1])

                # # Check if each embedding is already present in the index before adding it
                # for emb in embeddings:
                #     distances, indices = index.search(np.array([emb]), k=1)
                #     if distances[0][0] < 1:
                #         print("Embedding already present in index")
                #     else:
                #         index.add(np.array([emb]))
                # Add the embeddings to the index
                index.add(embeddings)

                # Save the index to the file
                faiss.write_index(index, index_path)
            else:
                if os.path.isfile(custom_pii_path):
                    with open(custom_pii_path, "r+") as f:
                        tmp = f.read()
                        custom_texts = ast.literal_eval(tmp)
                else:
                    return render_template('index.html', custom_error=True ,result=result, test_input=test_input, matched=matched, score=score, active_tab=active_tab)
            test_file = request.files.getlist('test-file')
            test_input = extractText(test_file)

            results = search(test_input, custom_texts)
            for result in results:
                for item in result:
                    if item[1] > 0.85:
                        result='pii'
                        matched=item[0]
                        score=item[1]
                        test_input = test_input[0]
                        break
                    else:
                        result='non-pii'
                        continue
                break
            return render_template('index.html', custom_error=custom_error, result=result, test_input=test_input, matched=matched, score=score, active_tab=active_tab)
    else:
        result = ''
    return render_template('index.html', custom_error=custom_error, result=result, test_input=test_input, matched=matched, score=score, active_tab=active_tab)

def extractText(files_data):
    payload = {'isOverlayRequired': False,
               'apikey': ocrKey,
               'language': 'eng',
               }
    text = []

    for file_data in files_data:
        r = requests.post('https://api.ocr.space/parse/image',
                files={file_data.filename: file_data.read()},
                data=payload,
            )
        list = json.loads(r.content.decode())['ParsedResults']
        file_texts = ''
        for sample in list:
            file_texts=file_texts+sample['ParsedText']
        text.append(file_texts)
    return text

# Define the function to generate embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

# Define a function to perform search based on similarity score
def search(queries, texts, k=1):
    # Retrieve the Faiss index from the .index file
    index_retrieved = faiss.read_index(index_path)

    query_embeddings = [get_embedding(query) for query in queries]
    results = []
    for query_embedding in query_embeddings:
        distances, indices = index_retrieved.search(np.array([query_embedding]), k)
        result = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
        results.append(result)
    return results
