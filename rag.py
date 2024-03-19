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

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
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
    template=local_llm_prompt_template, input_variables=["context", "question"]
)
retriever = db.as_retriever(search_kwargs={"k": 10})
multi_retriever = MultiQueryRetriever.from_llm(llm=gpt4, retriever=db.as_retriever())
multi_retriever_local_llm = MultiQueryRetriever.from_llm(
    llm=local_llm, retriever=db.as_retriever()
)


def get_queries(query_string):
    # Using ChatOpenAI to get more queries using MultiQueryRetriever

    queries = multi_retriever.get_relevant_documents(query=query_string)
    queries_local_llm = multi_retriever_local_llm.get_relevant_documents(
        query=query_string
    )
    gpt4_query_list = []
    local_llm_query_list = []
    for query in queries:
        print("Query from MultiQueryRetriver is: ", query)
        gpt4_query_list.append(query.page_content)
    for query in queries_local_llm:
        print("Query from MultiQueryRetriver for Local LLM is: ", query)
        local_llm_query_list.append(query.page_content)
    return gpt4_query_list, local_llm_query_list


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(request: Request, query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    chain_type_kwargs_local_llm = {"prompt": local_llm_prompt}

    # Update the milvus vector store with the new documents from pubmed
    print("Going to load and store research papers for query: ", query)
    load_and_store_research_papers(query, 10)
    qa_gpt4 = RetrievalQAWithSourcesChain.from_chain_type(
        llm=gpt4,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    qa_meditron_llm = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs_local_llm,
        verbose=True,
    )
    gpt4_response = qa_gpt4(query)
    local_llm_response = qa_meditron_llm(query)
    gpt4_response_list, local_llm_response_list = get_queries(query)
    # Now we will get different queries from MultiQueryRetriever and get additional GPT4 and Local LLM Response

    answer = gpt4_response["answer"]
    # # source_document = gpt4_response["source_documents"][0].page_content
    # source_document = " "
    # for source_doc in gpt4_response["source_documents"]:
    #     print("The source document is: ", source_doc.page_content)
    #     source_document += "<br>"
    #     source_document += source_doc.page_content
    #     source_document += "<br>"
    doc = "GPT4: Source Document: List <br>"
    index = 0
    # get the base URL of the uviorn server
    base_url = request.base_url
    # Convert the base URL to string
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
    local_llm_answer = local_llm_response["result"]
    local_llm_source_document = " "
    for local_llm_source_doc in local_llm_response["source_documents"]:
        print(
            "The source document for Local LLM is: ",
            local_llm_source_doc.metadata["source"],
        )
        local_llm_source_document += "<br>"
        local_llm_source_document += (
            "<a href='"
            + base_url
            + local_llm_source_doc.metadata["source"]
            + "'>"
            + local_llm_source_doc.metadata["source"]
            + "</a>"
        )
        local_llm_source_document += "<br>"

    local_llm_doc = "Local LLM: Source Document: List <br>"
    index = 0
    for source_doc in local_llm_response["source_documents"]:
        local_llm_doc += "<br> Source Document: " + str(index) + "<br>"
        local_llm_doc += source_doc.metadata["source"]
        local_llm_doc += "<br>"
        index += 1

    for response in gpt4_response_list:
        print("The response from MultiQueryRetriever is: ", response)
        # Add a new line to the answer
        answer += "<br>"
        answer += "#####GPT4 Multi Query Retriever Response: Start ######"
        answer += "<br>"
        answer += response
        answer += "<br>"
        answer += "#####GPT4 Multi Query Retriever Response: End ######"
        answer += "<br>"
    for response in local_llm_response_list:
        print("The response from MultiQueryRetriever for Local LLM is: ", response)
        # Add a new line to the answer
        local_llm_answer += "<br>"
        local_llm_answer += "Local LLM Multi Query Retriever Response: Start ######"
        local_llm_answer += "<br>"
        local_llm_answer += response
        local_llm_answer += "<br>"
        local_llm_answer += "Local LLM Multi Query Retriever Response: End ######"
        local_llm_answer += "<br>"

    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": answer,
                "source_document": doc,
                "doc": doc,
                "local_llm_answer": local_llm_answer,
                "local_llm_source_document": local_llm_source_document,
                "local_llm_doc": local_llm_doc,
            }
        )
    )

    res = Response(response_data)
    return res


# Define the main function
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
