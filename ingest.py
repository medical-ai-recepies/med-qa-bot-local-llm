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

ssl._create_default_https_context = ssl._create_unverified_context
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
local_path = (
    "./models/meditron-7b.Q4_K_M.gguf"  # replace with your desired local file path
)
local_llm = "meditron-7b.Q4_K_M.gguf"
os.environ["OPENAI_API_KEY"] = "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"
local_llm = GPT4All(model=local_path, verbose=True)
embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)
# Initializing the LLMs so that we get multiple queries for the retriver to get more articles

gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=3500, verbose=True)
# Build a simple chatbot using the PubmedReader

print(embeddings)

loader = DirectoryLoader(
    "static/data/", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
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
    #drop_old=True,
    collection_name="vector_db",
)


def load_and_store_research_papers(query_string, no_of_documents):
    loader = PubMedLoader(query_string)
    documents = loader.load()

    # Iterate each document and store it in a file directory as PDF
    dir_name = query_string.replace(" ", "_")
    if not os.path.exists("static/data/" + dir_name):
        os.makedirs("static/data/" + dir_name)

    for document in documents:
        print("Storing document: ", document)
        # Get the document title from document.metadata["Title"]
        title = document.metadata["Title"]
        print("Doc Title: ", title)
        # If Title is dict and contains i then take the i as title
        if isinstance(title, dict):
            title = title["i"]
            # If title[i] is an array then take the first element as title
            if isinstance(title, list):
                title = title[0]
        title = title.replace(" ", "_")
        title = title.replace("/", "_")
        # Shortern title to be under 50 characters
        title = title[:50]
        print("Doc Title: ", title)

        # Ensure directory exists
        directory_path = os.path.join("static/data", dir_name)
        os.makedirs(directory_path, exist_ok=True)

        # Set the file path
        file_path = os.path.join(directory_path, title + ".pdf")

        # Create a PDF file with the content
        # c = canvas.Canvas(file_path, pagesize=letter)
        # c.setFont("Helvetica", 12)

        # # Assuming 'page_content' is a string. If it includes multiple lines, they need to be handled.
        # text_object = c.beginText(40, 750)  # Start writing from the top of the page
        # text_object.setFont("Helvetica", 12)
        # # set the border of the text
        # text_object.setTextOrigin(40, 750)

        # # Handle multiline text
        # # Split the document.page_content into a list of lines
        # for line in document.page_content.split("\n"):
        #     # Add the line to the text object
        #     text_object.textLine(line)
        # c.drawText(text_object)
        # # Add the title to the PDF
        # c.setFont("Helvetica-Bold", 16)
        # if "Title" in document.metadata:
        #     doc_title = document.metadata["Title"]
        #     c.drawString(40, 800, doc_title)
        # # Add the URL to the PDF
        # c.setFont("Helvetica", 12)
        # # Check if the URL exists in the metadata
        # if "URL" in document.metadata:
        #     c.drawString(40, 780, document.metadata["URL"])
        # # Add the DOI
        # c.save()
        # Going to use pdfkit
        # Create a String object from page content and append Title and Copy Right at the top of the page
        doc_page_content = (
            "Title: "
            + json.dumps(document.metadata["Title"])
            + "\n"
            + "Copy Right: "
            + document.metadata["Copyright Information"]
            + "\n"
            + "Published Date: "
            + document.metadata["Published"]
            + "\n"
            + "\n"
            + "\n"
            + document.page_content
            + "\n"
            + "\n"
        )
        options = {
            "page-size": "Letter",
            "margin-top": "0.75in",
            "margin-right": "0.75in",
            "margin-bottom": "0.75in",
            "margin-left": "0.75in",
            "encoding": "UTF-8",
            "custom-header": [("Accept-Encoding", "gzip")],
            "no-outline": None,
        }
        pdfkit.from_string(doc_page_content, file_path, options=options)
        print(f"PDF file created: {file_path}")
    pdf_loader = DirectoryLoader(
        "static/data/" + dir_name,
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
        #drop_old=True,
        collection_name="vector_db",
    )


# url_quadrant = "http://localhost:6333"
# qdrant = Qdrant.from_documents(
#   texts, embeddings, url=url, prefer_grpc=False, collection_name="vector_db"
# )


print("Vector DB Successfully Created!")
