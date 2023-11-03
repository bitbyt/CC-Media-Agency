import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
    )

client = chromadb.PersistentClient(path="./chroma")
chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_HTTP_PORT)

local_collection = client.get_collection(name="langchain")
remote_collection = chroma_client.get_collection(name="cc_collection", embedding_function=openai_ef)

# embeddings = local_collection.get()['embeddings']
documents = local_collection.get()['documents']
metadatas = local_collection.get()['metadatas']
ids = local_collection.get()['ids']

remote_ids = remote_collection.get(offset=2000, limit=5)['ids']
item = local_collection.get()['embeddings']


last_count = 377
current_count = 0

# for index in range(len(documents)):
#     if current_count > last_count:
#         remote_collection.add(
#             documents=documents[index],
#             metadatas=metadatas[index],
#             ids=ids[index],
#         )
#     current_count += 1

print(item)

# list = ["geeks", "for", "geeks"]
# for index in range(len(list)):
#     print(list[index])