from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
import requests
from bs4 import BeautifulSoup
# from playwright.sync_api import sync_playwright

search_tool = TavilySearch(
    max_results=2,
    topic="general",    
    time_range="week",
    # include_domains=None,
    # exclude_domains=None
)

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    # print(f"Search results: {search_docs}")
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
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
def website_scrape(url: str, question: str) -> str:
    """Scrapes a website and returns the text.
    Args:
        url (str): the URL to the website to scrape.
    Returns:
        str: The text of the website.
    """

    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)
    #     page = browser.new_page()
    #     page.goto(url)
    #     html_content = page.content()
    #     browser.close()

    url = "https://en.wikipedia.org/wiki/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the infobox table
    # infobox = soup.find("table", {"class": "infobox"})

    # Extract text from the website
    text = soup.get_text()

    return text

if __name__ == "__main__":

    # Example usage
    query = "Mercedes Sosa"
    # results = website_scrape.invoke({"url":"https://en.wikipedia.org/wiki/", "question":query})

    results = wikipedia_search_html.invoke({"query": query})

    print(results)
    
    # Example usage of TavilySearch
    # search_results = search_tool.search("Python programming language")
    # print(search_results)

    
