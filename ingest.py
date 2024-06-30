import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader, PubMedLoader, PyPDFLoader
from langchain_community.llms import GPT4All
from langchain_openai import ChatOpenAI
import ssl
import pdfkit
import json

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Environment setup
os.environ["OPENAI_API_KEY"] = "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"


embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)
gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=4096, verbose=True)

vectorstore = None


def load_and_store_research_papers(query_string, no_of_documents=10):
    loader = PubMedLoader(query_string)
    documents = loader.load()

    # Create directory for storing documents
    dir_name = query_string.replace(" ", "_")
    directory_path = os.path.join("static/data", dir_name)
    os.makedirs(directory_path, exist_ok=True)

    for document in documents:
        # Get and sanitize the document title
        title = document.metadata.get("Title", "untitled")
        if isinstance(title, dict):
            title = title.get("i", title)
            if isinstance(title, list):
                title = title[0]
        title = title.replace(" ", "_").replace("/", "_")[:50]

        # Set the file path
        file_path = os.path.join(directory_path, title + ".pdf")

        # Create PDF content
        doc_page_content = (
            "Title: "
            + json.dumps(document.metadata.get("Title", ""))
            + "\nCopy Right: "
            + document.metadata.get("Copyright Information", "")
            + "\nPublished Date: "
            + document.metadata.get("Published", "")
            + "\n\n"
            + document.page_content
            + "\n\n"
        )

        # Create PDF file
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

    # Load and split documents
    pdf_loader = DirectoryLoader(
        directory_path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
    )
    documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # Initialize vectorstore
    vectorstore = Milvus.from_documents(
        texts,
        embeddings,
        connection_args={
            "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
            "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
        },
        collection_name="vector_db",
    )
    print("Vector DB Successfully Created!")


# Print embeddings for verification
print(embeddings)
