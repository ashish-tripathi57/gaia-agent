from typing import Dict, List, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from gaia_agent.common.tools import wikipedia_search_html, website_scrape, web_search, visual_model, audio_model, run_python, excel_tool
from gaia_agent.common.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, create_react_agent
from gaia_agent.agents.plan_execute.state import AgentState
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class FinalAnswer(BaseModel):
    """Final answer to user."""
    final_answer: str = Field(
        description="Final answer to user."
    )

class Act(BaseModel):
    """Action to perform."""
    action: Union[FinalAnswer, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

tools = [web_search, wikipedia_search_html, website_scrape, visual_model, audio_model, run_python, excel_tool]

# Planner node - generates responses
def planner(state: AgentState, config: Dict):
    logger.info("Planner node processing")
    parser = JsonOutputParser(pydantic_object=Plan)
    current_dir = Path(__file__).parent

    tools_str = "\n\n---\n\n".join([f"Tool: {tool.name}\nDescription: {tool.description}" for tool in tools])
    
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )
    
    planner_prompt = load_prompt(current_dir / ".." / ".." / "prompts", "planner")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", planner_prompt + "\n\n{format_instructions}"),
        ]
    )

    planner_chain = prompt | llm | parser
    response = planner_chain.invoke({"question": state["question"], "tools": tools_str, "format_instructions": parser.get_format_instructions()})
    return {"plan": response["steps"]}

# React agent node - generates responses
def react_agent(state: AgentState, config: Dict):
    """Generate a response using the LLM."""
    try:
        plan = state["plan"]
        # plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""{task}"""

        logger.info(f"React agent task: {task}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0,
            max_tokens=None,
            timeout=60,  # Added a timeout
            max_retries=2,
        )

        agent_executor = create_react_agent(llm, tools, prompt="You are a helpful assistant. You will be given a task and a list of tools. Use the tools to complete the task.")
        agent_response = agent_executor.invoke({"messages": state["messages"] + [("user", task_formatted)]})
        logger.info(f"React agent response: {agent_response['messages'][-1].content}")
        return {
            "past_steps": [(f"Task: {task}", f"Response: {agent_response['messages'][-1].content}\n")],
            "messages": [HumanMessage(content=task_formatted), agent_response["messages"][-1]]
        }
    except Exception as e:
        logger.error(f"Error in react agent node: {e}")
        return {
            "error": f"React agent error: {str(e)}",
            "messages": [AIMessage(content="I encountered an error while generating a response. Please try again.")]
        }


def replanner(state: AgentState, config: Dict):
    logger.info("Replanner node processing")

    parser = JsonOutputParser(pydantic_object=Act)
    current_dir = Path(__file__).parent

    tools_str = "\n\n---\n\n".join([f"Tool: {tool.name}\nDescription: {tool.description}" for tool in tools])

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    replanner_prompt = load_prompt(current_dir / ".." / ".." / "prompts", "replanner")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", replanner_prompt + "\n\n{format_instructions}"),
        ]
    )

    replanner_chain = prompt | llm | parser
    response = replanner_chain.invoke({"question": state["question"], "tools": tools_str, "plan": state["plan"], "past_steps": state["past_steps"], "format_instructions": parser.get_format_instructions()})
    if "final_answer" in response["action"] and response["action"]["final_answer"]["final_answer"]:
        return {"final_answer": response["action"]["final_answer"]["final_answer"], "last_ai_message": response["action"]["final_answer"]["final_answer"]}
    else:
        return {"plan": response["action"]["steps"]}

# Should end node - returns END if a response is available, otherwise returns react_agent
def should_end(state: AgentState, config: Dict):
    if "final_answer" in state and state["final_answer"]:
        return "__end__"
    else:
        return "react_agent"