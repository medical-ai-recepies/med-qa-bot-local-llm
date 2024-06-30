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
from ingest_google_scholar import load_and_store_research_papers  # LineList
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Import uvicorn
import uvicorn
from langchain_openai import ChatOpenAI
from typing import List

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.vectorstores import Milvus

from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate
import os
import json
import openai
from QueryGoogleScholar import QueryGoogleScholar

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/PDFs", StaticFiles(directory="PDFs"), name="PDFs")

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

os.environ["OPENAI_API_KEY"] = (
    "sk-proj-Ykm8eQLEhrnSfCi5DLoFT3BlbkFJccWzeOYM9AfDSGd8NfNA"
)
gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=4096, verbose=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

url = "http://localhost:6333"

db = Milvus(
    embeddings,
    connection_args={
        "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
        "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
    },
    collection_name="vector_db_scholar",
)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["summaries", "question"],
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = db.as_retriever(search_kwargs={"k": 100})
chat_history = []

multi_gpt4_llm_chain = LLMChain(llm=gpt4, prompt=QUERY_PROMPT)
multi_retriever = MultiQueryRetriever(
    retriever=db.as_retriever(), llm_chain=multi_gpt4_llm_chain, parser_key="lines"
)


def get_queries(query_string):
    queries = multi_retriever.get_relevant_documents(query=query_string)
    gpt4_query_list = []
    for query in queries:
        print("Query from MultiQueryRetriever is: ", query)
        gpt4_query_list.append(query.page_content)
    return gpt4_query_list


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(request: Request, query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}

    print("Going to load and store research papers for query: ", query)
    load_and_store_research_papers(query)
    gpt4_response_list_answer = ""
    qa_gpt4 = RetrievalQAWithSourcesChain.from_chain_type(
        llm=gpt4,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    gpt4_response = qa_gpt4(query)
    gpt4_response_list = get_queries(query)

    answer = gpt4_response["answer"]
    doc = "GPT4: Source Document: List <br>"
    index = 0
    base_url = request.base_url
    base_url = str(base_url)
    print("The base URL is: ", base_url)

    for source_doc in gpt4_response["source_documents"]:
        doc += "<br> Source Document: " + str(index) + "<br>"
        doc += (
            "<a href='"
            + base_url
            + source_doc.metadata["source"]
            + "'>"
            + source_doc.metadata["source"]
            + "</a>"
        )
        doc += "<br>"
        index += 1

    for response in gpt4_response_list:
        print("The response from MultiQueryRetriever is: ", response)
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += (
            "#####GPT4 Multi Query Retriever Response: Start ######"
        )
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += response
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += (
            "#####GPT4 Multi Query Retriever Response: End ######"
        )
        gpt4_response_list_answer += "<br>"

    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": answer,
                "source_document": doc,
                "doc": doc,
                "gpt4_response_list_answer": gpt4_response_list_answer,
            }
        )
    )

    res = Response(response_data)
    return res


@app.post("/continue_chat")
async def continue_chat(request: Request, query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}

    qa_gpt4 = RetrievalQAWithSourcesChain.from_chain_type(
        llm=gpt4,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )

    chat_history.append(query)
    context = "\n".join(chat_history)
    refined_query = f"{context}\n{query}"

    gpt4_response = qa_gpt4(refined_query)
    gpt4_response_list = get_queries(refined_query)

    answer = gpt4_response["answer"]
    gpt4_response_list_answer = ""
    doc = "GPT4: Source Document: List <br>"
    index = 0
    base_url = request.base_url
    base_url = str(base_url)
    print("The base URL is: ", base_url)

    for source_doc in gpt4_response["source_documents"]:
        doc += "<br> Source Document: " + str(index) + "<br>"
        doc += (
            "<a href='"
            + base_url
            + source_doc.metadata["source"]
            + "'>"
            + source_doc.metadata["source"]
            + "</a>"
        )
        doc += "<br>"
        index += 1

    for response in gpt4_response_list:
        print("The response from MultiQueryRetriever is: ", response)
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += (
            "#####GPT4 Multi Query Retriever Response: Start ######"
        )
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += response
        gpt4_response_list_answer += "<br>"
        gpt4_response_list_answer += (
            "#####GPT4 Multi Query Retriever Response: End ######"
        )
        gpt4_response_list_answer += "<br>"

    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": answer,
                "source_document": doc,
                "doc": doc,
                "gpt4_response_list_answer": gpt4_response_list_answer,
            }
        )
    )

    res = Response(response_data)
    return res


# Define the main function
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
