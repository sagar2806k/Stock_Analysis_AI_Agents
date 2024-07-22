from crewai import Crew
from textwrap import dedent
from agents import StockAnalysisAgents
from tasks import IndianStockAnalysisTasks
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

class FinancialCrew:
    def __init__(self, company):
        self.company = company
        self.agents = StockAnalysisAgents()
        self.tasks = IndianStockAnalysisTasks()

    def run(self):
        research_analyst_agent = self.agents.research_analyst()
        financial_analyst_agent = self.agents.financial_analyst()
        investment_advisor_agent = self.agents.investment_advisor()

        research_task = self.tasks.research(research_analyst_agent, self.company)
        financial_task = self.tasks.financial_analysis(financial_analyst_agent)
        filings_task = self.tasks.filings_analysis(financial_analyst_agent)
        recommend_task = self.tasks.recommend(investment_advisor_agent)

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
    company = input(dedent("""
        What is the company you want to analyze?
    """).strip())

    financial_crew = FinancialCrew(company)
    result = financial_crew.run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)
    
    with open("report.txt", "w") as f:
        f.write(result)
