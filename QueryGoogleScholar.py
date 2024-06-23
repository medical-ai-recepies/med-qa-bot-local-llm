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
from pydantic import BaseModel, Field
from typing import List
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper


class QueryGoogleScholar(BaseModel):
    query: str = Field(
        ...,
        title="Query",
        description="The query to run on Google Scholar",
    )

    def run(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        google_scholar = GoogleScholarAPIWrapper()
        google_scholar_query = GoogleScholarQueryRun(google_scholar)
        google_scholar_query.run_query(self.query)
        return google_scholar_query.results

    # Define main function to call the class


if __name__ == "__main__":
    query = QueryGoogleScholar(query="COVID-19")
    results = query.run()
    print(results)
