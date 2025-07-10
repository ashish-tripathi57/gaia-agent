from typing import Dict, Literal
from langchain_core.messages import AIMessage, HumanMessage
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
from langgraph.prebuilt import create_react_agent
from gaia_agent.agents.multi_agent.state import AgentState, Tasks
import inspect
from langgraph.types import Command
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _execute_agent_task(state: AgentState, config: Dict, agent_name: str, tools: list):
    """Common execution logic for agent tasks."""
    try:
        task = state["agent_tasks"][0].agent_task
        task_formatted = f"""{task}"""

        logger.info(f"{agent_name.replace('_', ' ').title()} task: {task}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0,
            max_tokens=None,
            timeout=60,
            max_retries=2,
        )

        prompt = f"You are a {agent_name} with access to tools.\n\n After you're done with your tasks, respond to the supervisor directly. Respond ONLY with the results of your work, do NOT include ANY other text."
        agent_executor = create_react_agent(llm, tools, prompt=prompt, name=agent_name)
        agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})

        final_response = f"{agent_name}'s task: {task}\n\n{agent_name}'s response: {agent_response['messages'][-1].content}"
        logger.info(f"{agent_name} response: {agent_response['messages'][-1].content}")

        return {
            "past_steps": [
                (
                    f"{agent_name}'s task: {task}",
                    f"{agent_name}'s Response: {agent_response['messages'][-1].content}\n",
                )
            ],
            "messages": [HumanMessage(content=final_response)],
        }
    except Exception as e:
        logger.error(f"Error in {agent_name} node: {e}")
        return {
            "error": f"{agent_name} error: {str(e)}",
            "messages": [
                AIMessage(
                    content="I encountered an error while generating a response. Please try again."
                )
            ],
        }


def web_search_agent(state: AgentState, config: Dict):
    """Web search agent can use tools to do web search and website scraping."""
    return _execute_agent_task(
        state, config, "web_search_agent", [web_search, website_scrape]
    )


def wikipedia_agent(state: AgentState, config: Dict):
    """Wikipedia agent can use wikipedia api to search for information."""
    return _execute_agent_task(
        state, config, "wikipedia_agent", [wikipedia_search_html]
    )


def visual_agent(state: AgentState, config: Dict):
    """Visual agent can use tools to process images."""
    return _execute_agent_task(state, config, "visual_agent", [visual_model])


def audio_agent(state: AgentState, config: Dict):
    """Audio agent can use tools to process audio."""
    return _execute_agent_task(state, config, "audio_agent", [audio_model])


def youtube_video_agent(state: AgentState, config: Dict):
    """Youtube video agent can use tools to process youtube videos."""
    return _execute_agent_task(state, config, "youtube_video_agent", [youtube_video_model])


def excel_agent(state: AgentState, config: Dict):
    """Excel agent can use tools to process excel files."""
    return _execute_agent_task(state, config, "excel_agent", [excel_tool])


def python_agent(state: AgentState, config: Dict):
    """Python agent can use tools to process python files."""
    return _execute_agent_task(state, config, "python_agent", [run_python])


def _collect_agents_with_docs():
    """Collect all agent functions from current module"""
    current_module = inspect.getmodule(inspect.currentframe())
    agents = []
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.endswith("_agent") and func.__module__ == current_module.__name__:
            docstring = inspect.getdoc(func) or "No docstring found."
            agents.append((name, docstring))
    return agents


# Cache the agent data
AGENT_DATA = _collect_agents_with_docs()
AGENT_NAMES = [name for name, _ in AGENT_DATA]


def get_agents():
    return AGENT_NAMES


def get_agents_with_docs():
    return AGENT_DATA


def supervisor(
    state: AgentState, config: Dict
) -> Command[
    Literal[
        *get_agents(),
        "validation_agent",
    ]
]:
    logger.info("Supervisor node processing")

    current_dir = Path(__file__).parent
    agents_str = "\n---\n".join(
        [
            f"Agent: {agent_name}\nDescription: {docstring}"
            for agent_name, docstring in get_agents_with_docs()
        ]
    )
    agents_str += "\n---\nAgent: validation_agent\nDescription: Once you have determined the final answer to the user's question, pass it to this agent. The answer must be fully self-contained, including all relevant information without relying on prior context.\n---\n"
    tasks_str = "---\n".join(f"{key}\n{val}" for key, val in state["past_steps"])
    tasks_str += "\n---\n"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )
    llm = llm.with_structured_output(Tasks)

    prompt_path = current_dir / ".." / ".." / "prompts"
    planner_prompt = load_prompt(prompt_path, "supervisor")
    prompt = ChatPromptTemplate.from_messages(
        [
            # ("human", planner_prompt + "\n\n{format_instructions}"),
            ("human", planner_prompt),
        ]
    )

    planner_chain = prompt | llm  # | parser
    response = planner_chain.invoke(
        {
            "question": state["question"],
            "past_steps": tasks_str,
            "agents": agents_str,
            # "format_instructions": parser.get_format_instructions(),
        }
    )

    agent_name, agent_task = (
        response.agent_tasks[0].agent_name,
        response.agent_tasks[0].agent_task,
    )
    return Command(
        goto=agent_name,
        update={
            "agent_tasks": response.agent_tasks,
            "last_ai_message": agent_task,
        },
    )
