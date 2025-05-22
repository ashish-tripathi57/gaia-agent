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
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from tools import wikipedia_search_html, website_scrape, web_search, visual_model, audio_model, run_python, data_tool
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

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
    tools = [web_search, wikipedia_search_html, website_scrape, visual_model, audio_model, run_python, data_tool]
    
    # Bind tools to the LLM
    chat_with_tools = llm.bind_tools(tools)

    # Define the state type with annotations
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        last_ai_message: Optional[str]
        question: Optional[str]
        final_answer: Optional[str]
        error: Optional[str]  # Added error field for tracking issues

    # Assistant node - generates responses
    def assistant(state: AgentState) -> Dict[str, Any]:
        """Generate a response using the LLM."""
        try:
            logger.info("Assistant node processing")
            # Check if the last message is from the assistant
            # prompt = f"Given the following question: {state['question']} Answer it using the tools available."
            # if state["planner_message"]:
            #     prompt += f"\nHere is a Plan to help you answer the user question: {state['planner_message']}"
            response = chat_with_tools.invoke(state["messages"])
            return {
                "messages": [response],
                "last_ai_message": response.content #if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
            }
        except Exception as e:
            logger.error(f"Error in assistant node: {e}")
            return {
                "error": f"Assistant error: {str(e)}",
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
    
    
    # Define your desired data structure.
    class AnswerTemplate(BaseModel):
        thought: str = Field(description="Thought process of the model")
        answer: str = Field(description="Final answer to the question")

    def validate_answer(state: AgentState) -> Dict[str, Any]:
        """Validate the final answer."""
        try:
            logger.info("Validating answer")
            
            llm_gemma = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                temperature=0,
                max_tokens=None,
                timeout=60,  # Added a timeout
                max_retries=2,
            )

            def escape_braces(text):
                return text.replace("{", "{{").replace("}", "}}")

            query = "You are given an interaction between human and AI agent. Format the AGENT ANSWER in json with the following keys: answer. Answer should be the final answer from the AGENT."
            
            # Set up a parser + inject instructions into the prompt template.
            parser = JsonOutputParser(pydantic_object=AnswerTemplate)
            prompt = PromptTemplate(
                        template=(
                            f"SYSTEM MESSAGE: {SYSTEM_MESSAGE}\n\n"
                            f"HUMAN QUERY: {escape_braces(state['question'])}\n\n"
                            f"AGENT ANSWER: {escape_braces(state['last_ai_message'])}\n\n" 
                            # "{format_instructions}\n{query}"
                            f"{query}"
                        ),
                        input_variables=["query"],
                        # partial_variables={"format_instructions": parser.get_format_instructions()},
                    )
            chain = prompt | llm_gemma | parser

            return {
                "final_answer": chain.invoke({"query": query})["answer"]
            }
        except Exception as e:
            logger.error(f"Error in validate_answer node: {e}")
            return {
                "error": f"Validation error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while validating the answer. Please try again.")]
            }

    # Build the graph
    builder = StateGraph(AgentState)

    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("validate_answer", validate_answer)
    # builder.add_node("handle_error", handle_error)

    # Define edges for the standard flow
    builder.add_edge(START, "assistant")

    # Conditional edges from assistant
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",  # Route to tools if needed
            # END: "END"  # Route to end if no tools needed
            END: "validate_answer"  # Route to validate_answer if no tools needed
        }
    )
    
    # From tools back to assistant to process tool results
    builder.add_edge("tools", "assistant")
    builder.add_edge("validate_answer", END)

    # Set up memory for conversation persistence
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # Generate and display the graph visualization
    # try:
    #     mermaid_code = graph.get_graph(xray=True).draw_mermaid()
    #     logger.info("Generated Mermaid diagram")

    #     with open("graph_diagram.mmd", "w") as f:
    #         f.write(mermaid_code)
    #         logger.info("Graph diagram saved as graph_diagram.mmd")
    # except Exception as e:
    #     logger.error(f"Error generating graph visualization: {e}")
    
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

    def __call__(self, question: str, task_id: str) -> str:
        """Process a question through the agent and return the response."""
        logger.info(f"Agent received question: {question[:50]}...")
        # Create a system message to guide the model's behavior
        system_message = SystemMessage(
            content=SYSTEM_MESSAGE
        )
        try:
            # Construct initial state
            question += f"\n task_id={task_id}"
            initial_state = {
                "messages": [system_message, HumanMessage(content=question)],
                # "last_ai_message": None,
                "question": question,
                "error": None
            }

            # Run the LangGraph
            final_state = self.graph.invoke(initial_state, self.config)
            # Check for errors in the final state
            if "error" in final_state and final_state["error"] is not None:
                logger.error(f"Error in final state: {final_state['error']}")
            
            final_answer = final_state.get("final_answer", None)

            if final_answer:
                # If a final answer is available, return it
                logger.info(f"Agent returning final answer: {final_answer}")
                return final_answer

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
    # response = agent("Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?")
    # response = agent({"question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.", "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44"})
    # response = agent(question="What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?", task_id="cabe07ed-9eca-40ea-8ead-410ef5e83f91")
    # response = agent(question="""Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3.

    # In your response, please only list the ingredients, not any measurements. So if the recipe calls for "a pinch of salt" or "two cups of ripe strawberries" the ingredients on the list would be "salt" and "ripe strawberries".

    # Please format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.task_id=99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3""", task_id="99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3")
    # response = agent(question="""Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(

    # Could you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order. task_id=1f975693-876d-457b-a649-393859e79bf3""", task_id="1f975693-876d-457b-a649-393859e79bf3")

    # response = agent(question="""What is the final numeric output from the attached Python code?task_id=f918266a-b3e0-4914-865d-4faa564f1aef""", task_id="f918266a-b3e0-4914-865d-4faa564f1aef")

    response = agent(question="""The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.task_id=7bd855d8-463d-4ed5-93ca-5fe35145f733""", task_id="7bd855d8-463d-4ed5-93ca-5fe35145f733")
    
    print(f"Response: {response}")