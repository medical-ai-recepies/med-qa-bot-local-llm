import requests
import concurrent.futures
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json
import pdfkit

# Importing to manage the PDF file size check for empty PDFs and pages check
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel, Field
from google_scholar_organic_query import GoogleScholarOrganicQuery


class QueryGoogleScholar(BaseModel):
    query: str = Field(
        ..., title="Query", description="The query to run on Google Scholar"
    )
    max_pages: int = Field(
        10, title="Max Pages", description="The maximum number of pages to fetch"
    )

    def run_query(self):
        start = time.time()
        profile_parser = GoogleScholarOrganicQuery()
        data = profile_parser.scrape_google_scholar_organic_results(
            query=self.query,
            api_key="bb49589c8703d72cbd466653fdcf0b6c1f50237804e3496e2f04219590b2ba57",  # https://serpapi.com/manage-api-key
            pagination=True,
            max_pages=self.max_pages,
            # other params
        )
        end = time.time()
        total = end - start
        total = round(total, 2)
        print(
            f"Time taken to run Google Scholar Query for search {self.query} is : {total} seconds"
        )
        return json.dumps(data, indent=4)

    def ensure_directory_exists(self):
        if not os.path.exists("PDFs"):
            os.makedirs("PDFs")

    def parse_html(self, html_content):
        return BeautifulSoup(html_content, "html.parser")

    def download_pdf_content(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join("PDFs", url.split("/")[-1])
            with open(file_path, "wb") as f:
                f.write(response.content)
            return "PDF downloaded successfully to: " + file_path
        return "Failed to download the PDF."

    def locate_and_download_pdf(self, url: str, title: str = None):
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._async_locate_and_download, url, title)
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred during the PDF download process: {e}")

        end = time.time()
        print(f"Time taken to download PDF: {round(end - start, 2)} seconds")

    def _async_locate_and_download(self, url, title, pages_to_fetch=5):
        if url.endswith(".pdf"):
            return self.download_pdf_content(url)

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        with webdriver.Chrome(options=options) as driver:
            driver.set_page_load_timeout(10)
            try:
                driver.get(url)
                driver.implicitly_wait(5)
            except Exception as e:
                return f"An error occurred while fetching the URL: {e}"

            html = driver.page_source

        soup = self.parse_html(html)
        if len(soup.get_text()) < 100:
            return f"Empty Page or Page has very few elements without any text. Skipping {title}"

        pdf_link = next(
            (
                link["href"]
                for link in soup.find_all("a")
                if "href" in link.attrs
                and "download" in link.text.lower()
                and link["href"].endswith(".pdf")
            ),
            None,
        )

        if pdf_link:
            return self.download_pdf_content(pdf_link)

        pdf_file_name = title if title else url.split("/")[-1]
        pdf_file_path = os.path.join("PDFs", f"{pdf_file_name}.pdf")
        try:
            pdfkit.from_url(url, pdf_file_path)
            reader = PdfReader(pdf_file_path)
            writer = PdfWriter()
            # Get the total pages in the PDF
            total_pages = len(reader.pages)
            if total_pages == 0:
                os.remove(pdf_file_path)
                return f"Empty PDF file. Skipping {title}"
            # Check if the PDF is empty and has no text
            text = reader.pages[0].extract_text()
            if len(text) < 100:
                os.remove(pdf_file_path)
                return f"Empty PDF file. Skipping {title}"
            # Check the size of PDF and if its more than 1.5 MB then extract only pages_to_fetch pages
            if os.path.getsize(pdf_file_path) > 1.5 * 1024 * 1024:
                for i in range(min(pages_to_fetch, total_pages)):
                    writer.add_page(reader.pages[i])
                writer.write(pdf_file_path)
                return f"Web page converted to PDF successfully and stored at: {pdf_file_path}"
            # Remove all BLANK pages from the PDF
            writer = PdfWriter()
            for i in range(total_pages):
                text = reader.pages[i].extract_text()
                if len(text) > 100:
                    writer.add_page(reader.pages[i])
            total_pages = len(writer.pages)
            return f"Web page converted to PDF successfully and stored at: {pdf_file_path} and total pages: {total_pages}"
        except Exception as e:
            if os.path.exists(pdf_file_path):
                os.remove(pdf_file_path)
            return f"An error occurred while writing file and removing the file {pdf_file_name}: {e}"

    def clean_directory(self):
        # Delete all the files in the directory
        print("Cleaning the PDFs directory")
        for file in os.listdir("PDFs"):
            file_path = os.path.join("PDFs", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"An error occurred while deleting the file: {e}")

    def extract_results(self, results, max_results_to_fetch):
        self.clean_directory()
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.locate_and_download_pdf, result["link"], result["title"]
                )
                for result in results[:max_results_to_fetch]
                if "link" in result and "title" in result
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    print(
                        "Parsing the HTML took too long and was stopped after 10 seconds."
                    )

        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        print(f"Total Time taken to download {max_results_to_fetch} PDFs: {total_time}")


if __name__ == "__main__":
    query = QueryGoogleScholar(query="COVID-19")
    print("Going to call Scholar API Directly")
    query = QueryGoogleScholar(
        query="Psoris and treatment options for skin", max_pages=20
    )
    results = query.run_query()
    results = json.loads(results)
    count = len(results)
    print(f"Total Results: {count}")
    max_results_to_fetch = min(100, count)
    print(
        f"Going to download PDFs up to {max_results_to_fetch} results from {count} results"
    )
    query.extract_results(results, max_results_to_fetch)
