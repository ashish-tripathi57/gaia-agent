from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
import uuid
import logging
from tools import wikipedia_search_html, website_scrape, web_search
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_graph():
    """Build and return a LangGraph for a conversational agent with tools."""
        
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    
    # Model configuration
    llm_model = "gemini-2.0-flash"  # Options: "gemma-3-27b-it", "gemini-2.0-flash-lite"

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    llm_gemma = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    # Define tools
    tools = [web_search, wikipedia_search_html, website_scrape]
    
    # Bind tools to the LLM
    chat_with_tools = llm.bind_tools(tools)

    # Define the state type with annotations
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        last_ai_message: Optional[str]
        human_message: Optional[str]
        planner_message: Optional[str]
        error: Optional[str]  # Added error field for tracking issues

    # Initialize the state
    def init(state: AgentState) -> Dict[str, Any]:
        """Extract the human message and initialize the state."""
        try:
            last_human = None
            for msg in reversed(state["messages"]):
                if last_human is None and isinstance(msg, HumanMessage):
                    last_human = msg
                    break
            
            # Add system message if it's the first message
            if len(state["messages"]) <= 1:
                return {
                    # "messages": [system_message],
                    "human_message": last_human.content if last_human else "",
                }
            return {
                "human_message": last_human.content if last_human else "",
            }
        except Exception as e:
            logger.error(f"Error in init node: {e}")
            return {
                "error": f"Initialization error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error during initialization. Please try again.")]
            }

    # Assistant node - generates responses
    def assistant(state: AgentState) -> Dict[str, Any]:
        """Generate a response using the LLM."""
        try:
            logger.info("Assistant node processing")
            # Check if the last message is from the assistant
            # prompt = f"Given the following question: {state['human_message']} Answer it using the tools available."
            # if state["planner_message"]:
            #     prompt += f"\nHere is a Plan to help you answer the user question: {state['planner_message']}"
            return {
                "messages": [chat_with_tools.invoke(state["messages"])],
                "last_ai_message": state["messages"][-1].content if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
            }
        except Exception as e:
            logger.error(f"Error in assistant node: {e}")
            return {
                "error": f"Assistant error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while generating a response. Please try again.")]
            }
        
    # Planner node - generates responses
    def planner(state: AgentState) -> Dict[str, Any]:
        """Create an action plan to answer user's query."""
        try:
            logger.info("Planner node processing")
            tools_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            prompt = f"Given you have access to these tools: {tools_info} Create a plan to answer the following question: {state['human_message']}"
            return {
                "messages": [HumanMessage(llm_gemma.invoke(prompt).content)],
            }
        except Exception as e:
            logger.error(f"Error in Planner node: {e}")
            return {
                "error": f"Planner error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while generating a response. Please try again.")]
            }

    # Error handling node
    def handle_error(state: AgentState) -> Dict[str, Any]:
        """Handle errors in the graph execution."""
        error_msg = state.get("error", "Unknown error")
        logger.error(f"Handling error: {error_msg}")
        return {
            "messages": [AIMessage(content=f"I apologize, but I encountered an error: {error_msg}. Please try again or rephrase your question.")]
        }

    # Build the graph
    builder = StateGraph(AgentState)

    # Define nodes
    # builder.add_node("init", init)
    # builder.add_node("planner", planner)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    # builder.add_node("handle_error", handle_error)

    # Define edges for the standard flow
    # builder.add_edge(START, "init")
    # builder.add_edge("init", "planner")
    # builder.add_edge("planner", "assistant")
    builder.add_edge(START, "assistant")


    # Conditional edges from assistant
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",  # Route to tools if needed
            END: END  # Route to end if no tools needed
        }
    )
    
    # From tools back to assistant to process tool results
    builder.add_edge("tools", "assistant")
    
    # Error handling edges
    # builder.add_conditional_edges(
    #     "init",
    #     lambda x: "error" in x and x["error"] is not None,
    #     {
    #         True: "handle_error",
    #         False: "assistant"
    #     }
    # )
    
    # builder.add_edge("handle_error", END)

    # Set up memory for conversation persistence
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # Generate and display the graph visualization
    try:
        mermaid_code = graph.get_graph(xray=True).draw_mermaid()
        logger.info("Generated Mermaid diagram")

        with open("graph_diagram.mmd", "w") as f:
            f.write(mermaid_code)
            logger.info("Graph diagram saved as graph_diagram.mmd")
    except Exception as e:
        logger.error(f"Error generating graph visualization: {e}")
    
    return graph

class BasicAgent:
    """A simple agent that manages interaction with the LangGraph."""
    
    def __init__(self, graph=None):
        """Initialize the agent with a LangGraph."""
        if graph is None:
            self.graph = get_graph()
        else:
            self.graph = graph
        logger.info("BasicAgent with LangGraph initialized.")

        thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Agent thread ID: {thread_id}")

    def __call__(self, question: str) -> str:
        """Process a question through the agent and return the response."""
        logger.info(f"Agent received question: {question[:50]}...")
        # Create a system message to guide the model's behavior
        system_message = SystemMessage(
            content="""You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
        )
        try:
            # Construct initial state
            initial_state = {
                "messages": [system_message, HumanMessage(content=question)],
                "last_ai_message": None,
                "human_message": None,
                "error": None
            }

            # Run the LangGraph
            final_state = self.graph.invoke(initial_state, self.config)
            final_messages = final_state.get("messages", [])

            # Get the last AI message (if any)
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                    response = msg.content
                    logger.info(f"Agent returning AI response")
                    return response

            # Fallback if no AI message found
            fallback = "Sorry, I could not generate a response."
            logger.warning("Agent fallback response - no AI message found")
            return fallback
            
        except Exception as e:
            logger.error(f"Unhandled exception in agent: {e}", exc_info=True)
            return f"Sorry, I encountered an unexpected error: {str(e)}"

# Example usage
if __name__ == "__main__":
    agent = BasicAgent()
    # response = agent("How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.")
    # response = agent('.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI')
    response = agent("Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?")
    
    print(f"Response: {response}")