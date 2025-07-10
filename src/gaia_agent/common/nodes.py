from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)


# Define your desired data structure.
class AnswerTemplate(BaseModel):
    # thought: str = Field(description="Thought process of the model")
    final_answer: str = Field(description="Final answer to the question")


def validate_answer(state):
    """Validate the final answer."""
    try:
        logger.info("Validating answer")

        llm_gemma = ChatGoogleGenerativeAI(
            model="gemma-3-12b-it",
            temperature=0,
            max_tokens=None,
            timeout=60,  # Added a timeout
            max_retries=2,
        )

        def escape_braces(text):
            return text.replace("{", "{{").replace("}", "}}")

        query = "---\n\nYou are given a conversation between a human and an AI agent. Identify the final answer provided by the agent. Then, format that final answer according to the formatting rules described in the system message, but do not alter the content of the answer itself. Only apply formatting as instructed."

        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=AnswerTemplate)
        prompt = PromptTemplate(
            template=(
                f"SYSTEM MESSAGE: {state['system_message']}\n\n"
                f"HUMAN QUERY: {escape_braces(state['question'])}\n\n"
                f"AGENT ANSWER: {escape_braces(state['last_ai_message'])}\n\n"
                f"{query}\n\n"
                "{format_instructions}"
            ),
            input_variables=["format_instructions"],
            # partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm_gemma | parser
        final_answer = chain.invoke(
            {"format_instructions": parser.get_format_instructions()}
        )["final_answer"]
        logger.info(f"Final answer: {final_answer}")
        return {"final_answer": final_answer}
    except Exception as e:
        logger.error(f"Error in validate_answer node: {e}")
        return {
            "error": f"Validation error: {str(e)}",
            "messages": [
                AIMessage(
                    content="I encountered an error while validating the answer. Please try again."
                )
            ],
        }
