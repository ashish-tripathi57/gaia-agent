# LangGraph AI Agent - Hugging Face Agent Course Final Project

This project is the final hands-on challenge of the Hugging Face Agents Course. The mission: create a sophisticated AI agent capable of tackling real-world tasks from the GAIA benchmark - a rigorous evaluation framework designed to test AI assistants on complex scenarios requiring reasoning, multimodal understanding, web browsing, and advanced tool use.


## The Dataset: GAIA

Uses a set of 20 questions selected from the Level 1 validation set of GAIA

## Agents

### 1. ReAct-style Agent
A LangGraph-based agent that uses the ReAct (Reason + Act) approach. It makes decisions based on its internal reasoning and chooses tools like web search, Python, or YouTube analysis to gather information before answering.

### 2. Reflection Agent
An advanced version of the ReAct agent that includes a self-reflection step. After an initial answer, it evaluates its own reasoning and decides whether more tool use is needed before finalizing a response.

### 3. Plan-Execute Agent
Implements a Plan-Execute architecture using LangChain and LangGraph. The agent generates a step-by-step plan and executes it through a reasoning loop, validating the final answer.

### 4. Multi-Agent System
Implements a multi-agent architecture using LangGraph and LangChain. A central supervisor agent coordinates a group of specialized agents and dynamically routes tasks to them.