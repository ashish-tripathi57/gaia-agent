import logging
import uuid
import os
from dotenv import load_dotenv

load_dotenv()  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    """A simple agent that manages interaction with the LangGraph."""
    
    def __init__(self, graph, system_message):
        """Initialize the agent with a LangGraph."""
        self.graph = graph
        self.system_message = system_message
        logger.info("Agent with LangGraph initialized.")

        thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Agent thread ID: {thread_id}")

    def __call__(self, question: str, task_id: str) -> str:
        """Process a question through the agent and return the response."""
        logger.info(f"Agent received question: {question[:50]}...")
        try:
            # Construct initial state
            exists = any(fname.startswith(task_id + ".") for fname in os.listdir("../downloaded_files/"))
            if exists:
                logger.info("File exists. Adding task_id to question")
                question += f"\ntask_id={task_id}"
            
            initial_state = {
                "system_message": self.system_message,
                "question": question,
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