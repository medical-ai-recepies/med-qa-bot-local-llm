import os
import ssl
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI
from QueryGoogleScholar import QueryGoogleScholar

# Setting up SSL context to avoid verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Define the embeddings model
embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-Ykm8eQLEhrnSfCi5DLoFT3BlbkFJccWzeOYM9AfDSGd8NfNA"
)
# Initialize the LLM for querying
gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=4096, verbose=True)


# Function to load and store research papers
def load_and_store_research_papers(query_string: str):
    # Query Google Scholar
    google_query = QueryGoogleScholar(query=query_string, max_pages=10)
    results = google_query.run_query()
    results = json.loads(results)
    count = len(results)
    print(f"Total Results: {count}")

    # Determine the number of results to fetch
    max_results_to_fetch = min(100, count)
    print(
        f"Going to download PDFs up to {max_results_to_fetch} results from {count} results"
    )
    google_query.extract_results(results, max_results_to_fetch)

    # Load the downloaded PDFs
    pdf_loader = DirectoryLoader(
        "PDFs", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredPDFLoader
    )
    documents = pdf_loader.load()

    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Total Texts: {len(texts)}")

    # Store the texts in a vector store
    vectorstore = Milvus.from_documents(
        texts,
        embeddings,
        connection_args={
            "uri": "https://in03-95dd43464153a8a.api.gcp-us-west1.zillizcloud.com",
            "token": "c6284a1a3345e7686b37791a0eb6474aeb781cfbd5cc2df815efa32c9159d999027e95f155ef6ad7a7d5c6b95a5989a6687fa2b6",
        },
        collection_name="vector_db_scholar",
    )


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-XbUsVEqAZDhW9je90anIT3BlbkFJasvVa7C2sPaCd8cWC8ek"
)

# Example usage
if __name__ == "__main__":
    query = "machine learning in healthcare"
    load_and_store_research_papers(query)
    print("Vector DB Successfully Created!")
