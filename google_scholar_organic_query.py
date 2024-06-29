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
from serpapi import GoogleScholarSearch
from urllib.parse import urlsplit, parse_qsl
import itertools


class GoogleScholarOrganicQuery(SerpApiGoogleScholarOrganic):
    def scrape_google_scholar_organic_results(
        self,
        query: str,
        api_key: str = None,
        lang: str = "en",
        pagination: bool = False,
        max_pages: int = 10,
    ):
        """
        This function extracts all possible data from Google Scholar organic results. With or without pagination.

        Arguments:
        - query: search query
        - api_key: SerpApi api key, https://serpapi.com/manage-api-key
        - lang: language for the search. Default 'en'. More: https://serpapi.com/google-languages
        - pagination: True of False. Enables pagination from all pages. Default 'False'.

        Usage:

        from google_scholar_py.serpapi_backend.organic_results import SerpApiGoogleScholarOrganic

        parser = SerpApiGoogleScholarOrganic()
        data = parser.scrape_google_scholar_organic_results(
            query='minecraft',
            api_key='serpapi_api_key',
            pagination=True
        )

        print(data[0].keys()) # show available keys

        for result in data:
            print(result['title']) # and other data
        """

        if api_key is None:
            raise Exception(
                "Please enter a SerpApi API key to a `api_key` argument. https://serpapi.com/manage-api-key"
            )

        if api_key and query is None:
            raise Exception(
                "Please enter a SerpApi API key to a `api_key`, and a search query to `query` arguments."
            )

        params = {
            "api_key": api_key,  # serpapi api key: https://serpapi.com/manage-api-key
            "engine": "google_scholar",  # serpapi parsing engine
            "q": query,  # search query
            "hl": lang,  # language
            "start": 0,  # first page. Used for pagination: https://serpapi.com/google-scholar-api#api-parameters-pagination-start
        }

        search = GoogleScholarSearch(params)  # where data extracts on the backend

        if pagination:
            page_count = 1
            organic_results_data = []

            while True:
                results = search.get_dict()  # JSON -> Python dict

                if "error" in results:
                    print(results["error"])
                    break

                organic_results_data.append(results["organic_results"])

                # check for `serpapi_pagination` and then for `next` page
                if "next" in results.get("serpapi_pagination", {}):
                    search.params_dict.update(
                        dict(
                            parse_qsl(
                                urlsplit(results["serpapi_pagination"]["next"]).query
                            )
                        )
                    )
                    page_count += 1
                else:
                    break
                if page_count >= max_pages:
                    break

            # flatten list
            return list(itertools.chain(*organic_results_data))
        else:
            # remove page number key from the request parameters
            # parse first page only
            params.pop("start")

            search = GoogleScholarSearch(params)
            results = search.get_dict()

            if "error" in results:
                raise Exception(results["error"])

            return results["organic_results"]
