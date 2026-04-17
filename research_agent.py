"""Research Agent Pro with Human-in-the-Loop (HITL) middleware."""

import json
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import (
    ArxivAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

load_dotenv()

model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:cloud")
temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "research-session.db")
db_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)


# --- RESEARCH TOOLS ---

# Web Search - DDGs
ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the web for current information, news, and articles. "
                "Use this when you need up-to-date information or topics not covered "
                "by Wikipedia or academic papers. Input should be a search query."
)

# Wikipedia
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Query Wikipedia for encyclopedia articles. "
                "Best for factual information, historical events, and well-documented topics. "
                "Input should be a search term or topic."
)

# Arxiv
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
    description="Search arXiv for academic papers in physics, mathematics, computer science, "
                "quantitative biology, quantitative finance, and statistics. "
                "Best for scholarly research and technical topics. "
                "Input should be a search query for academic papers."
)


@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.

    Returns:
        str: Current datetime in ISO format (YYYY-MM-DD HH:MM:SS)
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = """You are a Research AI Agent, an expert at finding and synthesizing information.

Your capabilities:
- Search the web for current information using web_search
- Look up factual information on Wikipedia
- Find academic papers on arXiv
- Get the current datetime

Guidelines:
1. Always use the most appropriate tool for the query
2. For factual questions, start with Wikipedia
3. For current events or recent information, use web search
4. For academic/research topics, search arXiv
5. Synthesize information from multiple sources when appropriate
6. Provide clear, concise answers with source attribution
7. If a tool fails, try an alternative source

When answering:
- Be accurate and cite your sources
- If you cannot find information, clearly state that
- Ask clarifying questions if the query is ambiguous
"""

# --- MIDDLEWARE ---

human_in_loop = HumanInTheLoopMiddleware(
    interrupt_on={
        "web_search": {"allowed_decisions": ["approve", "edit", "reject"]},
        "wikipedia": {"allowed_decisions": ["approve", "edit", "reject"]},
        "arxiv": {"allowed_decisions": ["approve", "edit", "reject"]},
        "get_current_datetime": False,
    },
    description_prefix="Tool execution requires approval",
)

# --- AGENT SETUP ---

def create_research_agent():
    """Create a research agent with HITL and SQLite checkpointing."""
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    memory = SqliteSaver(conn=db_conn)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        checkpointer=memory,
        middleware=[human_in_loop],
        name="research_agent",
    )

    return agent


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_hitl_request(interrupts):
    if not interrupts:
        return None
    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    if isinstance(payload, dict):
        return payload
    return None


def _collect_human_decisions(hitl_request):
    decisions = []
    action_requests = hitl_request.get("action_requests", [])
    review_configs = hitl_request.get("review_configs", [])

    print("\nHITL checkpoint reached. Review each tool call:\n")

    for index, action in enumerate(action_requests, start=1):
        config = review_configs[index - 1] if index - 1 < len(review_configs) else {}
        allowed = config.get("allowed_decisions", ["approve", "edit", "reject"])

        print(f"[{index}] Tool: {action.get('name')}")
        print(f"Args: {json.dumps(action.get('args', {}), indent=2)}")
        if action.get("description"):
            print(f"Context: {action['description']}")
        print(f"Allowed decisions: {', '.join(allowed)}")

        while True:
            decision_type = input("Decision: ").strip().lower()
            if decision_type in allowed:
                break
            print("Invalid decision. Please choose one of the allowed options.")

        if decision_type == "approve":
            decisions.append({"type": "approve"})
            continue

        if decision_type == "reject":
            message = input("Rejection reason (optional): ").strip()
            decision = {"type": "reject"}
            if message:
                decision["message"] = message
            decisions.append(decision)
            continue

        edited_name = input(f"Edited tool name [{action.get('name')}]: ").strip()
        if not edited_name:
            edited_name = action.get("name")

        current_args = action.get("args", {})
        while True:
            edited_args_text = input(
                "Edited args JSON (press Enter to keep current args): "
            ).strip()
            if not edited_args_text:
                edited_args = current_args
                break
            try:
                parsed = json.loads(edited_args_text)
                if not isinstance(parsed, dict):
                    print("Args must be a JSON object.")
                    continue
                edited_args = parsed
                break
            except json.JSONDecodeError as err:
                print(f"Invalid JSON: {err}")

        decisions.append(
            {
                "type": "edit",
                "edited_action": {
                    "name": edited_name,
                    "args": edited_args,
                },
            }
        )

        print()

    return decisions


def run_query_with_hitl(agent, query, config):
    """Run one query and handle all HITL pauses/resumes until completion."""
    payload = {"messages": [HumanMessage(content=query)]}

    while True:
        final_response_text = None
        hitl_request = None

        for chunk in agent.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                hitl_request = _extract_hitl_request(chunk["__interrupt__"])
                break

            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                messages = update.get("messages", [])
                for message in messages:
                    if isinstance(message, AIMessage):
                        text = _content_to_text(message.content)
                        if text:
                            final_response_text = text

        if hitl_request:
            decisions = _collect_human_decisions(hitl_request)
            payload = Command(resume={"decisions": decisions})
            continue

        if final_response_text:
            print(f"Agent: {final_response_text}")
        else:
            print("Agent: No response generated.")
        return

def banner():
    """Print the agent banner."""
    print("Research Agent Pro (HITL)")
    print("=" * 40)
    print("Type your research query.")
    print("For sensitive tool calls, choose: approve / edit / reject.")
    print()


def main():
    """Main entry point for the HITL research agent CLI."""
    banner()
    agent = create_research_agent()
    config = {"configurable": {"thread_id": "session1"}}

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            run_query_with_hitl(agent, query, config)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()