import json
import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

class SearchTools:

    @tool("Search the internet")
    def search_internet(self, query):
        """Function to search the internet about a given topic and return relevant results."""
        return self.search(query)

    def search(self, query, top_result_to_return=3):
        """Private method to handle search requests."""
        url = "https://google.serper.dev/news"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'content-type': 'application/json'
        }

        response = requests.request("POST", url, data=payload, headers=headers)
        results = response.json().get('organic') or response.json().get('news')
        string = []
        for result in results[:top_result_to_return]:
            try:
                string.append('\n'.join([
                    f"Title: {result['title']}", f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}", "\n-----------------"]))
            except KeyError:
                pass
        content = "\n".join(string)
        
        return f"\nSearch result : {content}\n"
