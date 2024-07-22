from crewai import Task
from textwrap import dedent

class IndianStockAnalysisTasks:
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
                Description: A comprehensive summary of the latest news, any notable shifts in market sentiment, and potential impacts on the stock.
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
                Type: analysis
                Description: A clear assessment of the stock's financial standing, its strengths and weaknesses, and how it fares against its competitors in the current market scenario.
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
                Type: report
                Description: An expanded report highlighting significant findings from filings, including any red flags or positive indicators.
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
                Type: recommendation
                Description: A full, super detailed report providing a clear investment stance and strategy with supporting evidence.
            """)
        )

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a ₹10,000 commission!"
