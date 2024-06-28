from langchain_community.llms import GPT4All
from langchain_community.embeddings import AwaEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain.chains.retrieval_qa.base import RetrievalQA
from ingest import load_and_store_research_papers
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import uvicorn
import uvicorn
from langchain_openai import ChatOpenAI

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.vectorstores import Milvus


from langchain.prompts import PromptTemplate
import os
import json
import openai
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_path = (
    "./models/meditron-7b.Q4_K_M.gguf"  # replace with your desired local file path
)
local_llm = "meditron-7b.Q4_K_M.gguf"


config = {
    "max_new_tokens": 1024,
    "context_length": 5000,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}
os.environ["OPENAI_API_KEY"] = "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"
local_llm = GPT4All(model=local_path, verbose=True)
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=3500, verbose=True)


print("LLM Initialized....")


prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Answer: 
Only return the helpful answer below and nothing else.
Helpful answer: 
=============
{summaries}
=============
"""

local_llm_prompt_template = """ 
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Answer:
Only return the helpful answer below and nothing else.
Helpful answer:
============
{summaries}
============
"""

embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

url = "http://localhost:6333"

# client = QdrantClient(url=url, prefer_grpc=False)

# db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
db = Milvus(
    embeddings,
    connection_args={
        "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
        "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
    },
    collection_name="vector_db",
)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["summaries", "question"],
)


local_llm_prompt = PromptTemplate(
    template=local_llm_prompt_template, input_variables=["summaries", "question"]
)
retriever = db.as_retriever(search_kwargs={"k": 10})
multi_retriever = MultiQueryRetriever.from_llm(llm=gpt4, retriever=db.as_retriever())
multi_retriever_local_llm = MultiQueryRetriever.from_llm(
    llm=local_llm, retriever=db.as_retriever()
)
from langchain_community.llms import GPT4All
from langchain_community.embeddings import AwaEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain.chains.retrieval_qa.base import RetrievalQA
from ingest import load_and_store_research_papers
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import uvicorn
import uvicorn
from langchain_openai import ChatOpenAI

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.vectorstores import Milvus


from langchain.prompts import PromptTemplate
import os
import json
import openai
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_path = (
    "./models/meditron-7b.Q4_K_M.gguf"  # replace with your desired local file path
)
local_llm = "meditron-7b.Q4_K_M.gguf"


config = {
    "max_new_tokens": 1024,
    "context_length": 5000,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

# openai.api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["OPENAI_API_KEY"] = "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"
local_llm = GPT4All(model=local_path, verbose=True)
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=3500, verbose=True)


print("LLM Initialized....")


prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Answer: 
Only return the helpful answer below and nothing else.
Helpful answer: 
=============
{summaries}
=============
"""

local_llm_prompt_template = """ 
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Answer:
Only return the helpful answer below and nothing else.
Helpful answer:
============
{summaries}
============
"""

embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

url = "http://localhost:6333"

# client = QdrantClient(url=url, prefer_grpc=False)

# db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
db = Milvus(
    embeddings,
    connection_args={
        "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
        "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
    },
    collection_name="vector_db",
)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["summaries", "question"],
)


local_llm_prompt = PromptTemplate(
    template=local_llm_prompt_template, input_variables=["summaries", "question"]
)
retriever = db.as_retriever(search_kwargs={"k": 10})
multi_retriever = MultiQueryRetriever.from_llm(llm=gpt4, retriever=db.as_retriever())
multi_retriever_local_llm = MultiQueryRetriever.from_llm(
    llm=local_llm, retriever=db.as_retriever()
)

from langchain_community.llms import GPT4All
from langchain_community.embeddings import AwaEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain.chains.retrieval_qa.base import RetrievalQA
from ingest import load_and_store_research_papers
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import uvicorn
import uvicorn
from langchain_openai import ChatOpenAI

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.vectorstores import Milvus


from langchain.prompts import PromptTemplate
import os
import json
import openai
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_path = (
    "./models/meditron-7b.Q4_K_M.gguf"  # replace with your desired local file path
)
local_llm = "meditron-7b.Q4_K_M.gguf"
