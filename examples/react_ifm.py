"""A bounded live ReAct loop using IFM, with the fork's 3D trace viewer.

uv run --with openai --with python-dotenv python examples/react_ifm.py \
    --env-file ../practice_agent/.env
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI
from pydantic import BaseModel

from langgraphics import watch


class Decision(BaseModel):
    action: Literal["calculate", "finish"]
    operation: Literal["add", "multiply"]
    a: float
    b: float
    answer: str


class State(TypedDict):
    question: str
    observations: list[dict]
    decision: dict
    steps: int
    answer: str
    activity: str


def build_agent(client, model, max_steps=6):
    async def reason(state: State):
        if state["steps"] >= max_steps:
            raise RuntimeError(f"ReAct exceeded its {max_steps}-call budget")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a small ReAct arithmetic agent. Use one calculate action per arithmetic operation, "
                        "then finish after reading the observations. Never compute arithmetic yourself. "
                        "Available operations: add, multiply. Return only JSON with fields action "
                        "(calculate or finish), operation (add or multiply), a (number), b (number), answer (string). "
                        "For calculate leave answer empty; for finish use operation add, a=0, b=0."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": state["question"],
                            "observations": state["observations"],
                        }
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Decision",
                    "strict": True,
                    "schema": {
                        **Decision.model_json_schema(),
                        "additionalProperties": False,
                    },
                },
            },
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        content = response.choices[0].message.content or ""
        # Some compatible providers put prose around their JSON response.
        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character == "{":
                try:
                    value, _ = decoder.raw_decode(content[index:])
                    decision = Decision.model_validate(value)
                    break
                except (ValueError, TypeError):
                    continue
        else:
            raise RuntimeError("IFM returned no valid action JSON")
        print(f"Model step {state['steps'] + 1}: {decision.action}", flush=True)
        return {
            "decision": decision.model_dump(),
            "steps": state["steps"] + 1,
            "answer": decision.answer,
            "activity": (
                f"Call calculator: {decision.operation}({decision.a:g}, {decision.b:g})"
                if decision.action == "calculate"
                else f"Final answer: {decision.answer}"
            ),
        }

    async def calculate(state: State):
        decision = Decision.model_validate(state["decision"])
        value = (
            decision.a + decision.b
            if decision.operation == "add"
            else decision.a * decision.b
        )
        observation = {
            "operation": decision.operation,
            "a": decision.a,
            "b": decision.b,
            "result": value,
        }
        print(f"Tool observation: {json.dumps(observation)}", flush=True)
        return {
            "observations": [*state["observations"], observation],
            "activity": f"Observation: {decision.operation}({decision.a:g}, {decision.b:g}) = {value:g}",
        }

    builder = StateGraph(State)
    builder.add_node(
        "reason",
        reason,
        metadata={
            "langgraphics_role": "llm",
            "langgraphics_label": "Reason & choose action",
            "langgraphics_description": "IFM LLM · reads prompt + observations; chooses tool or final answer",
        },
    )
    builder.add_node(
        "calculate",
        calculate,
        metadata={
            "langgraphics_role": "tool",
            "langgraphics_label": "Calculator",
            "langgraphics_description": "Deterministic tool · add or multiply two numbers; returns an observation",
        },
    )
    builder.add_edge(START, "reason")
    builder.add_conditional_edges(
        "reason",
        lambda state: state["decision"]["action"],
        {"calculate": "calculate", "finish": END},
    )
    builder.add_edge("calculate", "reason")
    return builder.compile()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("../practice_agent/.env"))
    parser.add_argument(
        "--question",
        default="Use the calculator to compute 17 times 23, then add 19 to that result.",
    )
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--ws-port", type=int, default=8775)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    key = os.getenv("IFM_API_KEY")
    if not key:
        parser.error("Set IFM_API_KEY in the environment or the supplied --env-file")
    model = os.getenv("IFM_MODEL", "IFM/K2-Horizon-375B-A23B")
    async with AsyncOpenAI(
        api_key=key, base_url="https://api.ifm.ai/v1", max_retries=0, timeout=90
    ) as client:
        graph = watch(
            build_agent(client, model),
            port=args.port,
            ws_port=args.ws_port,
            open_browser=not args.no_open_browser,
            theme="dark",
        )
        print(f"Model: {model}", flush=True)
        print(
            f"3D viewer: http://localhost:{args.port}?ws_port={args.ws_port}&theme=dark",
            flush=True,
        )
        try:
            # astream keeps the local viewer alive after completion. The server also
            # retains this run so late connections and page reloads can replay it.
            async for result in graph.astream(
                {
                    "question": args.question,
                    "observations": [],
                    "decision": {},
                    "steps": 0,
                    "answer": "",
                    "activity": "",
                },
                config={"recursion_limit": 20},
                stream_mode="values",
            ):
                pass
            print(f"Answer: {result['answer']}", flush=True)
            print("Viewer remains available. Press Ctrl+C to stop.", flush=True)
            await asyncio.Event().wait()
        finally:
            await graph.shutdown()
            graph.http_server.server_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
