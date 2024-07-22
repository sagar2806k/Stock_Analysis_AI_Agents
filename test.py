import json
import os
from playwright.sync_api import sync_playwright
from langchain.tools import tool
from unstructured.partition.html import partition_html
from crewai import Agent, Task

class BrowserTools:

    @tool("Scrape Website content")
    def scrape_and_summarize_website(website, config=None):
        """Useful to scrape and summarize website content"""
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(website)
            html = page.content()
            browser.close()

        elements = partition_html(text=html)
        content = "\n\n".join([str(el) for el in elements])
        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]

        summaries = []
        for chunk in content:
            agent = Agent(
                role="Principal Researcher",
                goal="Do amazing research and summaries based on the content you are working with",
                backstory="You are a principal researcher at a big company and you need to do research about a given topic.",
                allow_delegation=False
            )
            task = Task(
                agent=agent,
                description=f'Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
            )
            summary = task.execute()
            summaries.append(summary)
        
        return "\n\n".join(summaries)

from langchain.tools import tool
class CalculatorTools():

  @tool("Make a calculation")
  def calculate(operation):
    """Useful to perform any mathematical calculations, 
    like sum, minus, multiplication, division, etc.
    The input to this tool should be a mathematical 
    expression, a couple examples are `200*7` or `5000/2*10`
    """
    return eval(operation)
  
import json
import os

import requests
from langchain.tools import tool


class SearchTools():
  @tool("Search the internet")
  def search_internet(query):
    """Useful to search the internet 
    about a a given topic and return relevant results"""
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)

  @tool("Search news on the internet")
  def search_news(query):
    """Useful to search news about a company, stock or any other
    topic and return relevant results"""""
    top_result_to_return = 4
    url = "https://google.serper.dev/news"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['news']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)
  
import os
import requests

from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from unstructured.partition.html import partition_html

class IndianStockTools:
    @tool("Search Quarterly Results")
    def search_quarterly(data):
        """
        Useful to search information from the latest quarterly results for a given stock.
        The input to this tool should be a pipe (|) separated text of length two, representing the stock ticker you are interested and what question you have from it.
        For example, `RELIANCE|what was last quarter's revenue`.
        """
        stock, ask = data.split("|")
        api_endpoint = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock}&apikey={os.environ['FINANCIAL_DATA_API_KEY']}"
        response = requests.get(api_endpoint)
        if response.status_code != 200:
            return "Sorry, I couldn't retrieve the data. Please check if the ticker is correct."
        
        data = response.json()
        if 'quarterlyEarnings' not in data:
            return "No quarterly earnings data available for this stock."

        latest_quarter = data['quarterlyEarnings'][0]
        answer = IndianStockTools.__embedding_search(latest_quarter, ask)
        return answer

    @tool("Search Annual Report")
    def search_annual(data):
        """
        Useful to search information from the latest annual report for a given stock.
        The input to this tool should be a pipe (|) separated text of length two, representing the stock ticker you are interested, what question you have from it.
        For example, `RELIANCE|what was last year's revenue`.
        """
        stock, ask = data.split("|")
        api_endpoint = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock}&apikey={os.environ['FINANCIAL_DATA_API_KEY']}"
        response = requests.get(api_endpoint)
        if response.status_code != 200:
            return "Sorry, I couldn't retrieve the data. Please check if the ticker is correct."
        
        data = response.json()
        if 'annualEarnings' not in data:
            return "No annual earnings data available for this stock."

        latest_annual = data['annualEarnings'][0]
        answer = IndianStockTools.__embedding_search(latest_annual, ask)
        return answer

    def __embedding_search(data, ask):
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

  
from crewai import Agent
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"
)
class StockAnalysisAgents():
  def financial_analyst(self):
    return Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        IndianStockTools.search_annual,
        IndianStockTools.search_quarterly
      ],
      llm=llm
    )

  def research_analyst(self):
    return Agent(
      role='Staff Research Analyst',
      goal="""Being the best at gather, interpret data and amaze
      your customer with it""",
      backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        YahooFinanceNewsTool(),
        IndianStockTools.search_annual,
        IndianStockTools.search_quarterly
      ],
      llm=llm
  )

  def investment_advisor(self):
    return Agent(
      role='Private Investment Advisor',
      goal="""Impress your customers with full analyses over stocks
      and completer investment recommendations""",
      backstory="""You're the most experienced investment advisor
      and you combine various analytical insights to formulate
      strategic investment advice. You are now working for
      a super important customer you need to impress.""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        YahooFinanceNewsTool()
      ],
      llm=llm
    )
  
from crewai import Task
from textwrap import dedent

class IndianStockAnalysisTasks():
    def research(self, agent, company):
        return Task(
            description=dedent(f"""
                Collect and summarize recent news articles, press
                releases, and market analyses related to the stock and
                its industry in India.
                Pay special attention to any significant events, market
                sentiments, and analysts' opinions. Also include upcoming 
                events like earnings and others.
    
                Your final answer MUST be a report that includes a
                comprehensive summary of the latest news, any notable
                shifts in market sentiment, and potential impacts on 
                the stock.
                Also make sure to return the stock ticker.
                
                {self.__tip_section()}
    
                Make sure to use the most recent data as possible.
    
                Selected company by the customer: {company}
            """),
            agent=agent,
            expected_output=dedent("""
                Type: report
                Description: A comprehensive summary of the latest news, market sentiment, and potential impacts on the stock, including the stock ticker.
            """)
        )
        
    def financial_analysis(self, agent):
        return Task(
            description=dedent(f"""
                Conduct a thorough analysis of the stock's financial
                health and market performance in India. 
                This includes examining key financial metrics such as
                P/E ratio, EPS growth, revenue trends, and 
                debt-to-equity ratio. 
                Also, analyze the stock's performance in comparison 
                to its industry peers and overall market trends.

                Your final report MUST expand on the summary provided
                but now including a clear assessment of the stock's
                financial standing, its strengths and weaknesses, 
                and how it fares against its competitors in the current
                market scenario.{self.__tip_section()}

                Make sure to use the most recent data possible.
            """),
            agent=agent,
            expected_output=dedent("""
                Type: financial_report
                Description: A detailed financial analysis report including key metrics, comparison with industry peers, and overall market trends.
            """)
        )

    def filings_analysis(self, agent):
        return Task(
            description=dedent(f"""
                Analyze the latest annual reports and other relevant filings 
                from SEBI, BSE, or NSE for the stock in question.
                Focus on key sections like Management's Discussion and
                Analysis, financial statements, insider trading activity, 
                and any disclosed risks.
                Extract relevant data and insights that could influence
                the stock's future performance.

                Your final answer must be an expanded report that now
                also highlights significant findings from these filings,
                including any red flags or positive indicators for
                your customer.
                {self.__tip_section()}        
            """),
            agent=agent,
            expected_output=dedent("""
                Type: filings_report
                Description: An expanded report highlighting significant findings from the latest filings, including red flags and positive indicators.
            """)
        )

    def recommend(self, agent):
        return Task(
            description=dedent(f"""
                Review and synthesize the analyses provided by the
                Financial Analyst and the Research Analyst.
                Combine these insights to form a comprehensive
                investment recommendation for the Indian stock market.
                
                You MUST Consider all aspects, including financial
                health, market sentiment, and qualitative data from
                the filings.

                Make sure to include a section that shows insider 
                trading activity, and upcoming events like earnings.

                Your final answer MUST be a recommendation for your
                customer. It should be a full super detailed report, providing a 
                clear investment stance and strategy with supporting evidence.
                Make it pretty and well formatted for your customer.
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output=dedent("""
                Type: investment_recommendation
                Description: A comprehensive investment recommendation report, including financial health, market sentiment, and supporting evidence.
            """)
        )

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a ₹10,000 commission!"

  
from crewai import Crew
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()

class FinancialCrew:
  def __init__(self, company):
    self.company = company

  def run(self):
    agents = StockAnalysisAgents()
    tasks = IndianStockAnalysisTasks()

    research_analyst_agent = agents.research_analyst()
    financial_analyst_agent = agents.financial_analyst()
    investment_advisor_agent = agents.investment_advisor()

    research_task = tasks.research(research_analyst_agent, self.company)
    financial_task = tasks.financial_analysis(financial_analyst_agent)
    filings_task = tasks.filings_analysis(financial_analyst_agent)
    recommend_task = tasks.recommend(investment_advisor_agent)

    crew = Crew(
      agents=[
        research_analyst_agent,
        financial_analyst_agent,
        investment_advisor_agent
      ],
      tasks=[
        research_task,
        financial_task,
        filings_task,
        recommend_task
      ],
      verbose=True
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
  print("## Welcome to Financial Analysis Crew")
  print('-------------------------------')
  company = input(
    dedent("""
      What is the company you want to analyze?
    """))
  
  financial_crew = FinancialCrew(company)
  result = financial_crew.run()
  print("\n\n########################")
  print("## Here is the Report")
  print("########################\n")
  print(result)