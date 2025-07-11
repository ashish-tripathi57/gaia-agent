# LangGraph-based Agents for the GAIA Benchmark
### Hugging Face Agent Course Final Project

This project is the final hands-on challenge of the Hugging Face Agents Course. The mission: create a sophisticated AI agent capable of tackling real-world tasks from the GAIA benchmark - a rigorous evaluation framework designed to test AI assistants on complex scenarios requiring reasoning, multimodal understanding, web browsing, and advanced tool use.


## The Dataset: GAIA

Uses a set of 20 questions selected from the Level 1 validation set of GAIA

## Agents

This project implements various types of agents, each with a unique architecture for tackling complex tasks.

### 1. ReAct-style Agent
A LangGraph-based agent that uses the ReAct (Reason + Act) approach. It makes decisions based on its internal reasoning and chooses tools like web search, Python, or YouTube analysis to gather information before answering.

### 2. Reflection Agent
An advanced version of the ReAct agent that includes a self-reflection step. After an initial answer, it evaluates its own reasoning and decides whether more tool use is needed before finalizing a response.

### 3. Plan-Execute Agent
Implements a Plan-Execute architecture using LangChain and LangGraph. The agent generates a step-by-step plan and executes it through a reasoning loop, validating the final answer.

### 4. Multi-Agent System
Implements a multi-agent architecture using LangGraph and LangChain. A central supervisor agent coordinates a group of specialized agents and dynamically routes tasks to them.

## Project Structure

The project is organized into the following modules under `src/gaia_agent/`:

### `agents/`

This is the core directory containing the implementation of the different agent architectures.

*   **`base.py`**: Defines a simple `Agent` base class that manages the interaction with the LangGraph graph, handling state, configuration, and the execution flow.

*   **`react/`**: Implements a ReAct-style agent.

*   **`reflection/`**: Implements an agent with a reflection step.

*   **`plan_execute/`**: Implements a plan-and-execute agent.

*   **`multi_agent/`**: Implements a multi-agent system with a supervisor.

### `common/`

This directory contains modules with code shared across the different agents.

*   **`tools.py`**: This is a crucial module that defines all the tools available to the agents. These tools include:
    *   `excel_tool`: For querying Excel files.
    *   `run_python`: For executing Python scripts.
    *   `audio_model`: For processing audio files.
    *   `visual_model`: For processing images.
    *   `youtube_video_model`: For analyzing YouTube videos.
    *   `web_search`: For searching the web.
    *   `wikipedia_search_html`: For searching Wikipedia and getting HTML content.
    *   `website_scrape`: For scraping websites.

*   **`nodes.py`**: Contains the `validate_answer` node, which is used by all agents to format the final answer according to the user's request.

*   **`prompts.py`**: Provides a `load_prompt` function to load prompt templates from text files.

### `prompts/`

This directory contains the text files for the prompts used by the agents.

*   `planner.txt`: The prompt for the planner in the plan-and-execute agent.
*   `reflection.txt`: The prompt for the reflection step in the reflection agent.
*   `replanner.txt`: The prompt for the replanner in the plan-and-execute agent.
*   `supervisor.txt`: The prompt for the supervisor in the multi-agent system.