from playwright.sync_api import sync_playwright
from langchain.tools import tool
from unstructured.partition.html import partition_html
from crewai import Agent, Task

class BrowserTools:
    @tool("Scrape website content")
    def scrape_and_summarize_website(self, website):
        """Scrape and summarize website content."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(website)
            content = page.content()
            browser.close()

        elements = partition_html(text=content)
        content = "\n\n".join([str(el) for el in elements])
        content_chunks = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        summaries = []
        for chunk in content_chunks:
            agent = Agent(
                role='Principal Researcher',
                goal='Do amazing research and summaries based on the content you are working with',
                backstory="You're a Principal Researcher at a big company and you need to do research about a given topic.",
                allow_delegation=False
            )
            task = Task(
                agent=agent,
                description=f'Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
            )
            summary = task.execute()
            summaries.append(summary)
        return "\n\n".join(summaries[:5000])
