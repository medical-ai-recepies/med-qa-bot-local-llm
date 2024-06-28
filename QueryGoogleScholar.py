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
from google_scholar_py import SerpApiGoogleScholarOrganic
import json
import requests
from bs4 import BeautifulSoup
import pdfkit
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


class QueryGoogleScholar(BaseModel):
    query: str = Field(
        ...,
        title="Query",
        description="The query to run on Google Scholar",
    )

    def run(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        # pass the SERP API key here
        os.environ["SERP_API_KEY"] = (
            "bb49589c8703d72cbd466653fdcf0b6c1f50237804e3496e2f04219590b2ba57"
        )
        google_scholar = GoogleScholarAPIWrapper()
        google_scholar_query = GoogleScholarQueryRun(api_wrapper=google_scholar)
        results = google_scholar_query.run(self.query)
        return results

    def ensure_directory_exists(self):
        if not os.path.exists("PDFs"):
            os.makedirs("PDFs")

    def run_query(self):
        start = time.time()
        profile_parser = SerpApiGoogleScholarOrganic()
        data = profile_parser.scrape_google_scholar_organic_results(
            query=self.query,
            api_key="bb49589c8703d72cbd466653fdcf0b6c1f50237804e3496e2f04219590b2ba57",  # https://serpapi.com/manage-api-key
            pagination=True,
            # other params
        )
        end = time.time()
        total = end - start
        print(
            f"Time taken to run Google Scholar Query for search {self.query} is : {total} seconds"
        )
        return json.dumps(data, indent=4)

    def locate_and_download_pdf(self, url: str, title: str = None):
        # Make a request to the webpage
        # Add start and end time for the method
        start = time.time()

        try:
            # Setup Chrome options
            options = Options()
            options.add_argument("--headless")  # Enables headless mode
            options.add_argument("--window-size=1920,1080")  # Sets the window size
            options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration
            options.add_argument(
                "--no-sandbox"
            )  # Bypass OS security model (necessary on some systems)
            options.add_argument(
                "--disable-dev-shm-usage"
            )  # Overcomes limited resource problems

            # Instantiate a webdriver with options
            driver = webdriver.Chrome(options=options)
            # Open the URL with Selenium
            driver.get(url)

            # Optionally, add some wait here if the page needs time to load JavaScript content
            driver.implicitly_wait(
                2
            )  # You might need to adjust the waiting time based on page load

            # Get the page source
            html = driver.page_source
            driver.quit()  # Make sure to close the driver after task completion
            # Ensure the PDFs directory exists
            self.ensure_directory_exists()
            # Use BeautifulSoup to parse the fetched HTML
            soup = BeautifulSoup(html, "html.parser")
            # Check to see if its a Empty Page or the page has very few elements without any text
            if len(soup.get_text()) < 100:
                print(
                    f"Empty Page or Page has very few elements without any text Skipping {title}"
                )
                return
            # Search for a download link that possibly links to a PDF
            pdf_link = None
            for link in soup.find_all("a"):
                if (
                    "href" in link.attrs
                    and "download" in link.text.lower()
                    and link["href"].endswith(".pdf")
                ):
                    pdf_link = link["href"]
                    break

            if pdf_link:
                # Download the PDF
                pdf_response = requests.get(pdf_link)
                if pdf_response.status_code == 200:
                    pdf_file_path = os.path.join("PDFs", pdf_link, "_downloaded.pdf")
                    with open(pdf_file_path, "wb") as f:
                        f.write(pdf_response.content)
                    print("PDF downloaded successfully to:", pdf_file_path)
                else:
                    print("Failed to download the PDF.")
            else:
                # Convert the entire page to PDF as a fallback using wkhtmltopdf
                # name the pdf file to that of the title of the page
                try:
                    pdf_file_name = title if title != None else url.split("/")[-1]
                    pdf_file_path = os.path.join("PDFs", pdf_file_name + ".pdf")
                    pdfkit.from_url(url, pdf_file_path)
                    print(
                        "Web page converted to PDF successfully and stored at:",
                        pdf_file_path,
                    )
                except Exception as e:
                    print(
                        f"An error occurred: while writing file and removing the file {pdf_file_name}",
                        e,
                    )
                    os.remove(pdf_file_path)
        except Exception as e:
            print("An error occurred:", e)

        end = time.time()
        total = end - start
        print(f"Time taken to download PDF: {total} seconds")

    # Connect to

    # Define main function to call the class


if __name__ == "__main__":
    query = QueryGoogleScholar(query="COVID-19")
    # results = query.run()
    # print(results)
    print("Going to call Scholar API Directly")
    query = QueryGoogleScholar(query="Oriental Herbs for Skin care")
    results = query.run_query()
    # Convert it into results array
    results = json.loads(results)
    count = len(results)
    print(f"Total Results: {count}")
    # print(results)
    # Iterate results object and extract the link attribute from the result json
    print("Going to download PDFs")
    for result in results:
        # See if result has link attribute
        if "link" not in result:
            continue
        if "title" not in result:
            continue
        print(result["link"])
        print(result["title"])
        # print(result["publication_info"])
        query.locate_and_download_pdf(result["link"], result["title"])

    # print(results)
