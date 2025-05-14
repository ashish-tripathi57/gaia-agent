from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=2,
    topic="general",    
    time_range="week",
    # include_domains=None,
    # exclude_domains=None
)