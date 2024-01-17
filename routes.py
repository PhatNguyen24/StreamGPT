from flask import Flask, render_template, request, Response
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

import os
import openai
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. 
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

def prompt(query):
     docs = docsearch.similarity_search(query, k=4) # tìm kiếm 4 văn bản tương tự 
     prompt = gen_prompt(docs, query)
     return prompt

def stream(input_text):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You're an assistant."},
            {"role": "user", "content": f"{prompt(input_text)}"},
        ], stream=True, temperature=0.5)
        for line in completion:
            if 'content' in line['choices'][0]['delta']:
                yield line['choices'][0]['delta']['content']  #tương tự return, khác cái là trả về hết vòng lặp thì mới thoát khỏi hàm
                # print(line['choices'][0]['delta']['content'],end='', flush=True)

@app.route('/completion', methods=['GET', 'POST'])
def completion_api():
    if request.method == "POST":
        data = request.form
        input_text = data['input_text']
        return Response(stream(input_text), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)

