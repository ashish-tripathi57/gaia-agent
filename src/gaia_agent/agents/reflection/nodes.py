from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
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
from gaia_agent.common.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from gaia_agent.agents.reflection.state import AgentState
from pathlib import Path
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


def reflection(state: AgentState, config: Dict):
    logger.info("Reflection node processing")
    current_dir = Path(__file__).parent

    # Data model
    class ReflectionOutput(BaseModel):
        """Reflection output."""

        ready_to_answer: bool = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
        reasoning: str = Field(
            description="Reasoning and possible feedback for the agent"
        )

    # LLM with function call
    parser = JsonOutputParser(pydantic_object=ReflectionOutput)

    reflection_prompt = load_prompt(current_dir / ".." / ".." / "prompts", "reflection")
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-12b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )
    # execution_trace = get_buffer_string(state["messages"])
    filtered_messages = [
        msg for msg in state["messages"] if not isinstance(msg, ToolMessage)
    ]
    execution_trace = get_buffer_string(filtered_messages)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", reflection_prompt + "\n\n{format_instructions}"),
        ]
    )

    reflection_chain = prompt | llm | parser
    response = reflection_chain.invoke(
        {
            "execution_trace": execution_trace,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    ready_to_answer = response["ready_to_answer"]
    reasoning = response["reasoning"]
    if ready_to_answer:
        logger.info("Agent is ready to answer")
        return {"ready_to_answer": ready_to_answer}
    else:
        logger.info("Agent is not ready to answer")
        logger.info(f"Reasoning: {reasoning[:100]}...")
        return {
            "messages": [HumanMessage(content=reasoning)],
            "ready_to_answer": ready_to_answer,
        }


def ready_to_answer(state: AgentState, config: Dict):
    if state["ready_to_answer"]:
        return "validate_answer"
    else:
        return "assistant"
