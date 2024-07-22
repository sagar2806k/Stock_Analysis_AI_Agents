from crewai import Agent
from Tools.browser import BrowserTools
from Tools.calculator import CalculatorTools
from Tools.search_tool import SearchTools
from Tools.sec_tool import IndianStockTools
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the language model
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

class StockAnalysisAgents:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

    def financial_analyst(self):
        return Agent(
            role="You are the best Financial Analyst in INDIA.",
            goal="Impress all the customers with your financial data and market trends analysis.",
            backstory="""The most seasoned financial analyst with 
            lots of expertise in stock market analysis and investment
            strategies that is working for a super important customer.""",
            verbose=True,
            tools=[
                BrowserTools().scrape_and_summarize_website,
                SearchTools().search_internet,
                CalculatorTools().calculate,
                IndianStockTools().search_annual,
                IndianStockTools().search_quarterly,
            ],
            llm=self.llm,
           
        )
    
    def research_analyst(self):
        return Agent(
            role="You are the best Research Analyst in INDIA.",
            goal="Help users research for information about stock market trends.",
            backstory="""The most seasoned research analyst with a deep understanding of the stock market and its trends.
            Known as the BEST research analyst, you're skilled in sifting through news, company announcements, 
            and market sentiments. Now you're working on a super important customer.""", 
            verbose=True,
            tools=[
                BrowserTools().scrape_and_summarize_website,
                SearchTools().search_internet,
                IndianStockTools().search_annual,
                IndianStockTools().search_quarterly,
            ],
            llm=self.llm
           
        )
    
    def investment_advisor(self):
        return Agent(
            role="Private Investment Advisor",
            goal="""Help users make informed investment decisions using their financial data and market trends analysis,
            Impress your customers with full analyses over stocks and complete investment recommendations.""",
            backstory="""The most seasoned investment advisor with a deep understanding of the stock market and its trends. 
            Known as the BEST investment advisor, you're skilled in analyzing financial data, 
            making informed investment decisions, and providing guidance to your clients.
            Now you're working on a super important customer. You're the most experienced investment advisor
            and you combine various analytical insights to formulate
            strategic investment advice. You are now working for
            a super important customer you need to impress.""", 
            verbose=True,
            tools=[
                BrowserTools().scrape_and_summarize_website,
                SearchTools().search_internet,
                CalculatorTools().calculate,
                IndianStockTools().search_annual,
                IndianStockTools().search_quarterly,
            ],
            llm=self.llm
            
        )
