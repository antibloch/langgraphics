"""Minimal real ReAct loop: model -> ToolNode -> model, with exactly three tools.

uv run --group react python examples/three_tool_react_ifm.py \
    --env-file ../practice_agent/.env

IFM structured decisions are adapted to standard AIMessage.tool_calls. All tool
execution and loop routing are performed by LangGraph, with actual observations
fed back to the model. No planner or additional workflow stages are inserted.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from complex_react_ifm import (
    ActionDecision,
    calculator,
    convert_units,
    lookup_product,
    openai_messages,
    parse_action,
)
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai import AsyncOpenAI

from langgraphics import watch

TOOLS = [lookup_product, calculator, convert_units]
DEFAULT_PROMPT = (
    "Look up the Summit kit. Use calculator to find the price of 4 kits using its "
    "returned unit price. Then use convert_units to convert the weight of one "
    "kit from kg to g. Call those three tools in order, one per turn. Read each "
    "observation before choosing the next action, then answer with the total "
    "price and weight in grams."
)


class State(MessagesState):
    steps: int


def build_agent(client, model):
    schema = ActionDecision.model_json_schema()
    schema["$defs"]["ActionCall"]["properties"]["name"]["enum"] = [
        t.name for t in TOOLS
    ]

    async def agent(state):
        if state.get("steps", 0) >= 6:
            raise RuntimeError("Exceeded the six model-call budget")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Fulfill the request with the available tools. Never invent observations. "
                        "Choose one tool per turn, read its actual observation, and then choose "
                        "the next action. Use calculator for arithmetic. After all requested "
                        "tools return, finish with a concise answer based on their results. "
                        "Return JSON: action (tools or finish), calls "
                        "([{name, arguments_json}]), answer (string). arguments_json must be "
                        "a JSON-encoded argument object. For tools leave answer empty; for "
                        "finish use an empty calls list. Only these tools exist: "
                        + json.dumps([convert_to_openai_tool(t) for t in TOOLS])
                    ),
                },
                {
                    "role": "user",
                    "content": "Actual conversation and tool observations (JSON): "
                    + json.dumps(openai_messages(state["messages"])),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ActionDecision",
                    "strict": True,
                    "schema": schema,
                },
            },
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        decision = parse_action(response.choices[0].message.content or "")
        if (decision.action == "tools") != bool(decision.calls):
            raise ValueError("Tool actions require calls; finish requires no calls")
        if len(decision.calls) > 1:
            raise ValueError("This minimal demo executes one tool per model turn")
        calls = []
        for call in decision.calls:
            if call.name not in {t.name for t in TOOLS}:
                raise ValueError(f"Tool is not registered: {call.name}")
            calls.append(
                {
                    "name": call.name,
                    "args": json.loads(call.arguments_json),
                    "id": "call_" + uuid4().hex[:16],
                    "type": "tool_call",
                }
            )
        print(
            "Model action: " + (calls[0]["name"] if calls else "final answer"),
            flush=True,
        )
        return {
            "messages": [AIMessage(content=decision.answer, tool_calls=calls)],
            "steps": state.get("steps", 0) + 1,
        }

    builder = StateGraph(State)
    builder.add_node(
        "agent",
        agent,
        metadata={
            "langgraphics_role": "llm",
            "langgraphics_label": "ReAct agent",
        },
    )
    builder.add_node(
        "tools",
        ToolNode(TOOLS, handle_tool_errors=True),
        metadata={
            "langgraphics_role": "executor",
            "langgraphics_label": "Tool executor",
        },
    )
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("../practice_agent/.env"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--ws-port", type=int, default=8795)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    key = os.getenv("IFM_API_KEY")
    if not key:
        parser.error("IFM_API_KEY is missing")
    model = os.getenv("IFM_MODEL", "IFM/K2-Horizon-375B-A23B")
    async with AsyncOpenAI(
        api_key=key, base_url="https://api.ifm.ai/v1", timeout=90, max_retries=0
    ) as client:
        viewer = watch(
            build_agent(client, model),
            port=args.port,
            ws_port=args.ws_port,
            open_browser=not args.no_open_browser,
            theme="dark",
        )
        print(
            f"Viewer: http://localhost:{args.port}/?ws_port={args.ws_port}&theme=dark",
            flush=True,
        )
        print("Prompt: " + args.prompt, flush=True)
        try:
            result = await viewer.ainvoke(
                {"messages": [HumanMessage(content=args.prompt)], "steps": 0},
                config={"recursion_limit": 20},
            )
            print("Answer: " + result["messages"][-1].content, flush=True)
        except Exception as error:  # noqa: BLE001 - preserve failed trace for inspection
            print(f"Run failed: {type(error).__name__}: {error}", flush=True)
        finally:
            if args.trace_file:
                viewer.export_trace(args.trace_file)
                print(f"Trace: {args.trace_file}", flush=True)
        print("Viewer remains open.", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await viewer.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
