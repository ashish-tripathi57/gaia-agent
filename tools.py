from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os
# from playwright.sync_api import sync_playwright

search_tool = TavilySearch(
    max_results=2,
    # topic="general",    
    # time_range="week",
    # include_domains=None,
    # exclude_domains=None
)


# Define your desired data structure.
class ImprovedQuery(BaseModel):
    query1: str = Field(description="An improved query version 1")
    # query2: str = Field(description="An improved query version 2")
    # query3: str = Field(description="An improved query version 3")
    # query4: str = Field(description="An improved query version 4")
    # query5: str = Field(description="An improved query version 5")

@tool
def web_search(query: str) -> str:
    """Search the web for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    
    alternative_queries = generate_improved_queries(query)
    alternative_queries['query'] = query
    # import pdb;pdb.set_trace()
    search_results = []
    for key, val in alternative_queries.items():
        search_results.append(search_tool.invoke(val))
    # print(f"Search results: {search_results} \n type: {type(search_results)}")
    return {"search_results": str(search_results)}

def generate_improved_queries(query: str) -> str:
    """
    Generate one improved versions of a given search query using a language model.

    Args:
        query (str): The original search query to be improved.

    Returns:
        str: A JSON object containing five improved query versions.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    
    llm_gemma = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ImprovedQuery)
    prompt = PromptTemplate(
        template="Give 1 improved version of this search query: {query}\n.\n{format_instructions}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm_gemma | parser

    return chain.invoke({"query": query})


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    # print(f"Search results: {search_docs}")
    formatted_search_docs = "\n\n---\n\n".join(
        [
            # f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}


@tool
def wikipedia_search_html(query: str) -> str:
    """
    Search Wikipedia for a given query, retrieve the corresponding page's HTML content,
    clean it by removing unnecessary elements (such as styles, scripts, references, infoboxes, etc.),
    and return a simplified HTML string containing only the main content.

    Args:
        query (str): The search query for the Wikipedia page.

    Returns:
        str: Cleaned HTML string of the Wikipedia page's main content, or an empty string if not found.
    """
    
    # Step 1: Get Wikipedia HTML
    page = wikipedia.page(query)
    html = page.html()

    # Step 2: Parse HTML
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output")
    # content_div = soup.find("table", class_="wikitable")
    if not content_div:
        return ""

    # Step 3: Find all tags to remove (style, script, sup, infobox, etc.)
    to_decompose = []
    for tag in content_div.find_all():
        tag_classes = tag.get("class", [])
        if (
            tag.name in ["style", "script", "sup"]
            or any(cls in ["infobox", "navbox", "reference"] for cls in tag_classes)
        ):
            to_decompose.append(tag)

    # Remove them after collecting
    for tag in to_decompose:
        tag.decompose()

    # Step 4: Unwrap all tags except allowed ones
    allowed_tags = {"ul", "li", "table", "tr", "td", "th"}
    to_unwrap = [tag for tag in content_div.find_all() if tag.name not in allowed_tags]

    for tag in to_unwrap:
        tag.unwrap()

    # Step 5: Return cleaned HTML string
    return str(content_div)


@tool
def website_scrape(url: str) -> str:
    """
    Scrape the given website URL and return all extracted text content.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The plain text content extracted from the website's HTML.
    """

    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)
    #     page = browser.new_page()
    #     page.goto(url)
    #     html_content = page.content()
    #     browser.close()

    # url = "https://en.wikipedia.org/wiki/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the infobox table
    # infobox = soup.find("table", {"class": "infobox"})

    # Extract text from the website
    text = soup.get_text()

    return text

if __name__ == "__main__":

    # Example usage
    # query = "Mercedes Sosa"
    # results = website_scrape.invoke({"url":"https://en.wikipedia.org/wiki/", "question":query})
    # results = wikipedia_search_html.invoke({"query": query})
    # results = wiki_search.invoke({"query": "Wikipedia featured articles promoted in november 2016"})
    # results = web_search.invoke({"query": "featured article dinosaur english wikipedia november 2016"})
    results = website_scrape.invoke({"url":"https://en.wikipedia.org/wiki/Wikipedia:Featured_article_candidates/Giganotosaurus/archive1"})
    print(results)
    
    # Example usage of TavilySearch
    # search_results = search_tool.search("Python programming language")
    # print(search_results)

    
