from typing import Dict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from gaia_agent.common.tools import (
    wikipedia_search_html,
    website_scrape,
    web_search,
    visual_model,
    audio_model,
    youtube_video_model,
    run_python,
    excel_tool,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from gaia_agent.agents.react.state import AgentState
import logging

logger = logging.getLogger(__name__)


# Assistant node - generates responses
def assistant(state: AgentState, config: Dict):
    """Generate a response using the LLM."""
    try:
        logger.info("Assistant node processing")

        tools = [
            web_search,
            wikipedia_search_html,
            website_scrape,
            visual_model,
            audio_model,
            youtube_video_model,
            run_python,
            excel_tool,
        ]
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0,
            max_tokens=None,
            timeout=60,  # Added a timeout
            max_retries=2,
        )

        messages = []
        init = False
        if len(state["messages"]) == 0:
            messages = [
                SystemMessage(content=state["system_message"]),
                HumanMessage(content=state["question"]),
            ]
            init = True

        # Bind tools to the LLM
        chat_with_tools = llm.bind_tools(tools)
        response = chat_with_tools.invoke(messages if init else state["messages"])
        messages.append(response)
        logger.info(f"Assistant response: {response.content[:50]}...")
        return {
            "messages": messages,
            "last_ai_message": response.content,  # if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
        }
    except Exception as e:
        logger.error(f"Error in assistant node: {e}")
        return {
            "error": f"Assistant error: {str(e)}",
            "messages": [
                AIMessage(
                    content="I encountered an error while generating a response. Please try again."
                )
            ],
        }


def get_tool_node(state: AgentState, config: Dict):
    return ToolNode(
        [
            web_search,
            wikipedia_search_html,
            website_scrape,
            visual_model,
            audio_model,
            youtube_video_model,
            run_python,
            excel_tool,
        ]
    )
