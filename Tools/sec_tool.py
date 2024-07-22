import os
import requests
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.partition.html import partition_html

class IndianStockTools:
    @tool("Search Quarterly Results")
    def search_quarterly(self, data):
        """
        Useful to search information from the latest quarterly results for a given stock.
        The input to this tool should be a pipe (|) separated text of length two, representing the stock ticker you are interested and what question you have from it.
        For example, `RELIANCE|what was last quarter's revenue`.
        """
        stock, ask = data.split("|")
        api_endpoint = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock}&apikey={os.getenv('FINANCIAL_DATA_API_KEY')}"
        response = requests.get(api_endpoint)
        if response.status_code != 200:
            return "Sorry, I couldn't retrieve the data. Please check if the ticker is correct."
        
        data = response.json()
        if 'quarterlyEarnings' not in data:
            return "No quarterly earnings data available for this stock."

        latest_quarter = data['quarterlyEarnings'][0]
        answer = self.__embedding_search(latest_quarter, ask)
        return answer

    @tool("Search Annual Report")
    def search_annual(self, data):
        """
        Useful to search information from the latest annual report for a given stock.
        The input to this tool should be a pipe (|) separated text of length two, representing the stock ticker you are interested, what question you have from it.
        For example, `RELIANCE|what was last year's revenue`.
        """
        stock, ask = data.split("|")
        api_endpoint = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock}&apikey={os.getenv('FINANCIAL_DATA_API_KEY')}"
        response = requests.get(api_endpoint)
        if response.status_code != 200:
            return "Sorry, I couldn't retrieve the data. Please check if the ticker is correct."
        
        data = response.json()
        if 'annualEarnings' not in data:
            return "No annual earnings data available for this stock."

        latest_annual = data['annualEarnings'][0]
        answer = self.__embedding_search(latest_annual, ask)
        return answer

    def __embedding_search(self, data, ask):
        content = "\n".join([f"{key}: {value}" for key, value in data.items()])
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([content])
        retriever = FAISS.from_documents(
            docs, SentenceTransformerEmbeddings()
        ).as_retriever()
        answers = retriever.get_relevant_documents(ask, top_k=4)
        answers = "\n\n".join([a.page_content for a in answers])
        return answers
