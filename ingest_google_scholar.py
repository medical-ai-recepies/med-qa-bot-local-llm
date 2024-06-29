import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader, PubMedLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant, Milvus
import ssl
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from langchain_community.llms import GPT4All
from langchain_openai import ChatOpenAI
import pdfkit
import json
import openai

ssl._create_default_https_context = ssl._create_unverified_context
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
local_path = (
    "./models/meditron-7b.Q4_K_M.gguf"  # replace with your desired local file path
)
local_llm = "meditron-7b.Q4_K_M.gguf"
# get the API key from the github secrets
# openai.api_key = os.getenv("OPENAI_API_KEY")
# assign it to the environment variable
# os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-XbUsVEqAZDhW9je90anIT3BlbkFJasvVa7C2sPaCd8cWC8ek"
)
# local_llm = GPT4All(model=local_path, verbose=True)
embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)
# Initializing the LLMs so that we get multiple queries for the retriver to get more articles

gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=5000, verbose=True)
# Build a simple chatbot using the PubmedReader

print(embeddings)

loader = DirectoryLoader(
    "PDFs", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
)
# pub_loader = PubMedLoader(search_query=search_query, max_results=10, show_progress=True)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
vectorstore = Milvus.from_documents(
    texts,
    embeddings,
    connection_args={
        "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
        "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
    },
    # drop_old=True,
    collection_name="vector_db_scholar",
)


def load_and_store_research_papers(query_string: str):
    google_query = QueryGoogleScholar(query=query_string, max_pages=10)
    results = google_query.run_query()
    results = json.loads(results)
    count = len(results)
    print(f"Total Results: {count}")
    max_results_to_fetch = min(100, count)
    print(
        f"Going to download PDFs up to {max_results_to_fetch} results from {count} results"
    )
    google_query.extract_results(results, max_results_to_fetch)

    pdf_loader = DirectoryLoader(
        "PDFs",
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader,
    )
    documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vectorstore = Milvus.from_documents(
        texts,
        embeddings,
        connection_args={
            "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
            "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
        },
        # drop_old=True,
        collection_name="vector_db_scholar",
    )


# url_quadrant = "http://localhost:6333"
# qdrant = Qdrant.from_documents(
#   texts, embeddings, url=url, prefer_grpc=False, collection_name="vector_db"
# )


print("Vector DB Successfully Created!")
