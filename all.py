from fastapi import FastAPI, Form, Request
import numpy as np
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient, utility
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import openai
import configparser
import requests
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create an instance of FastAPI
app = FastAPI()

# Set up the template directory
templates = Jinja2Templates(directory="templates")

# Milvus client setup
cfp = configparser.RawConfigParser()
milvus_client = MilvusClient(
    uri='https://in03-39925560e7df35f.serverless.gcp-us-west1.cloud.zilliz.com',
    token='a7897f67b9b06dbdbad6a04f493c23e5d36c007c200da4b8757ad70af40a1a990d89e9c4dfca6b693341240b6e41275f4068f67c'
)
#print(f"Connected to DB: {milvus_client.list_collections()}")

# Home route
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

# Load route for scraping data
@app.post("/load")
async def scrap_data(r: Request, url: str = Form(...)):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        h1 = [p.text for p in soup.find_all('h1')]
        h2 = [p.text for p in soup.find_all('h2')]
        h3 = [p.text for p in soup.find_all('h3')]
        h4 = [p.text for p in soup.find_all('h3')]
        div = [p.text for p in soup.find_all('div')]
        para = [p.text for p in soup.find_all('p')]
        tr = [p.text for p in soup.find_all('tr')]
        td = [p.text for p in soup.find_all('td')]
        table = tr + td
        headers = h1 + h2 + h3 + h4
        all_texts = headers + para + div + td

        # Clean the text
        all_texts = [line.replace('\n', '').replace('\t', '') for line in all_texts]

        # Embed the text
        def emb_text(text):                     
            return model.encode([text], normalize_embeddings=True).tolist()[0]

        data = []
        for i, line in enumerate(tqdm(all_texts, desc="Creating embeddings")):
            data.append({"id": i, "vector": emb_text(line), "text": line})

        embedding_dim = len(data[0]['vector'])
        collection_name = "rag_collection"

        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
        
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )

        # Insert data into Milvus
        insert_res = milvus_client.insert(collection_name=collection_name, data=data)

        return templates.TemplateResponse("index2.html", {'request': r})
    
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}")

# Query route
@app.api_route("/query", methods=["GET", "POST"])
async def query_dat(r: Request, ui: str = Form(None)):
    def emb_text(text):                             
        return model.encode([text], normalize_embeddings=True).tolist()[0]

    if r.method == "POST" and ui:        
        question = ui + ' ?'
        collection_name = "rag_collection"

        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[emb_text(question)],
            limit=3,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )

        retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
        context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])

        h_token = "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"
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm_client = InferenceClient(model=repo_id, token=h_token, timeout=120)

        PROMPT = """Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
                    <context> {context} </context> <question> {question} </question>"""

        prompt = PROMPT.format(context=context, question=question)
        answer = llm_client.text_generation(prompt, max_new_tokens=1000).strip()
        query_data = answer

    elif r.method == "GET" and "ui" in r.query_params:
        query_data = f"Data from GET request: {r.query_params['ui']}"
    else:
        query_data = None

    return templates.TemplateResponse("index2.html", {"request": r, "query_data": query_data})
