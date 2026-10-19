"""A real ToolNode ReAct agent: eight tools, parallel dispatch, and replanning.

uv run --group react python examples/complex_react_ifm.py --env-file ../practice_agent/.env
"""

import argparse
import ast
import asyncio
import json
import operator
import os
import statistics
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from langgraphics import watch


@tool
def lookup_product(name: str) -> dict:
    """Read a product's unit price (USD) and weight (kg) from the local demo catalog."""
    catalog = {
        "Summit kit": {"price_usd": 84, "weight_kg": 2.4},
        "River bottle": {"price_usd": 18, "weight_kg": 0.35},
        "Trail lamp": {"price_usd": 32, "weight_kg": 0.25},
    }
    match = next(
        (value for key, value in catalog.items() if key.lower() == name.lower()), None
    )
    return (
        {"product": name, **match}
        if match
        else {"error": "Unknown product", "available": list(catalog)}
    )


@tool
def search_notes(query: str) -> dict:
    """Search local demo store policy notes. Topics: shipping, discounts, currency."""
    notes = {
        "shipping": "Use shipping_quote with total catalog weight and road distance. Shipping is added after discounts.",
        "discount": "WELCOME10 discounts merchandise by 10%; shipping is excluded. Prices have no additional tax in this demo.",
        "currency": "All catalog, discount and shipping amounts are in USD.",
    }
    matches = {
        key: value
        for key, value in notes.items()
        if key in query.lower() or key == "currency"
    }
    return matches or notes


@tool
def calculator(expression: str) -> float:
    """Evaluate bounded numeric arithmetic: +, -, *, /, parentheses. No Python code or variables."""
    if len(expression) > 200:
        raise ValueError("Expression too long")
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node, depth=0):
        if depth > 20:
            raise ValueError("Expression too deep")
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and abs(node.value) <= 1e12
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            value = allowed[type(node.op)](
                visit(node.left, depth + 1), visit(node.right, depth + 1)
            )
            if abs(value) > 1e15:
                raise ValueError("Result too large")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(
                node.operand, depth + 1
            )
        raise ValueError("Only numeric arithmetic is supported")

    return round(float(visit(ast.parse(expression, mode="eval").body)), 6)


@tool
def summarize_numbers(values: list[float]) -> dict:
    """Return count, mean, minimum, maximum, and sum for up to 100 numeric values."""
    if not 1 <= len(values) <= 100:
        raise ValueError("Supply 1 to 100 values")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert mass between kg and g, or distance between km and m."""
    factors = {
        ("kg", "g"): 1000,
        ("g", "kg"): 0.001,
        ("km", "m"): 1000,
        ("m", "km"): 0.001,
    }
    factor = 1 if from_unit == to_unit else factors.get((from_unit, to_unit))
    if factor is None:
        raise ValueError("Supported conversions: kg/g, km/m")
    return {"value": round(value * factor, 6), "unit": to_unit}


@tool
def shipping_quote(weight_kg: float, distance_km: float) -> dict:
    """Quote demo shipping in USD: base $5 + $1.50/kg + $0.01/km."""
    if weight_kg < 0 or distance_km < 0:
        raise ValueError("Weight and distance must be nonnegative")
    return {
        "shipping_usd": round(5 + weight_kg * 1.5 + distance_km * 0.01, 2),
        "weight_kg": weight_kg,
        "distance_km": distance_km,
    }


@tool
def apply_discount(subtotal_usd: float, code: str) -> dict:
    """Apply WELCOME10 to merchandise; return discounted subtotal and amount saved in USD."""
    if code.upper() != "WELCOME10" or subtotal_usd < 0:
        raise ValueError("Use WELCOME10 with a nonnegative subtotal")
    return {
        "subtotal_usd": subtotal_usd,
        "saved_usd": round(subtotal_usd * 0.1, 2),
        "discounted_usd": round(subtotal_usd * 0.9, 2),
    }


@tool
def format_report(title: str, lines: list[str]) -> str:
    """Format a concise final purchase report from already verified observations. Returns text, writes no files."""
    return title + "\n" + "\n".join("- " + line for line in lines[:20])


TOOLS = [
    lookup_product,
    search_notes,
    calculator,
    summarize_numbers,
    convert_units,
    shipping_quote,
    apply_discount,
    format_report,
]
DEFAULT_QUESTION = (
    "Prepare a purchase report for 3 Summit kits and 2 River bottles shipped 450 km. "
    "Look up both products and shipping/discount notes. Use calculator for subtotal and total weight, "
    "summarize_numbers for the five individual unit prices, apply WELCOME10, convert total mass to grams, "
    "get a shipping_quote, and use calculator for the delivered total. Finally call format_report with "
    "the verified amounts and answer with that report. Batch independent tool calls when possible."
)


class ActionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal[
        "lookup_product",
        "search_notes",
        "calculator",
        "summarize_numbers",
        "convert_units",
        "shipping_quote",
        "apply_discount",
        "format_report",
    ]
    arguments_json: str


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tools", "finish"]
    calls: list[ActionCall]
    answer: str


def parse_action(content, decision_type=ActionDecision, call_type=ActionCall):
    for marker in ("</ifm|think>", "</think>"):
        if marker in content:
            content = content.rsplit(marker, 1)[-1]
    decoder = json.JSONDecoder()
    calls = []
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except (ValueError, TypeError):
            continue
        try:
            return decision_type.model_validate(value)
        except (ValueError, TypeError):
            pass
        try:
            calls.append(call_type.model_validate(value))
        except (ValueError, TypeError):
            pass
    if calls:
        # IFM may return one JSON call per line instead of an enclosing object.
        # Preserve only the calls it actually emitted, in their original order.
        return decision_type(action="tools", calls=calls, answer="")
    raise ValueError("No executable action JSON")


class State(MessagesState):
    steps: int
    activity: str
    report: str


def openai_messages(messages):
    result = []
    for message in messages:
        if isinstance(message, SystemMessage):
            result.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            result.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            entry = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"]),
                        },
                    }
                    for call in message.tool_calls
                ]
            result.append(entry)
        elif message.type == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
    return result


def build_agent(client, model, max_model_calls=10):
    async def prepare(state):
        if not any(
            isinstance(m, HumanMessage) and m.content for m in state["messages"]
        ):
            raise ValueError("Supply a user message")
        return {"steps": 0, "activity": "Validated user prompt"}

    async def planner(state):
        print("Calling IFM: planner", flush=True)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write at most five short bullet points planning tool use for this request. "
                        "Do not compute or invent tool results. Available tools: "
                        + ", ".join(t.name for t in TOOLS)
                    ),
                },
                {"role": "user", "content": state["messages"][-1].content},
            ],
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        plan = response.choices[0].message.content
        if not plan:
            raise RuntimeError("IFM returned an empty plan")
        return {
            "messages": [SystemMessage(content="Execution plan: " + plan)],
            "steps": 1,
            "activity": "Plan created from the user request",
        }

    async def agent(state):
        if state["steps"] >= max_model_calls:
            raise RuntimeError(f"Stopped at the {max_model_calls}-call budget")
        print(f"Calling IFM: action step {state['steps'] + 1}", flush=True)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Fulfill the user request using real tools. Never invent tool observations. "
                        "Use calculator for arithmetic and reuse tool results. Batch independent calls in one response. "
                        "Use every tool explicitly requested. After format_report returns, finish with its report and an empty calls list. "
                        "Return JSON with action tools or finish, calls as [{name, arguments_json}], and answer. "
                        "arguments_json must be a JSON-encoded argument object matching the tool schema. For tools action leave answer empty. "
                        "Exact argument keys: lookup_product(name), search_notes(query), calculator(expression), "
                        "summarize_numbers(values), convert_units(value, from_unit, to_unit), "
                        "shipping_quote(weight_kg, distance_km), apply_discount(subtotal_usd, code), format_report(title, lines). "
                        'Example call: {"name":"lookup_product","arguments_json":"{\\"name\\":\\"Summit kit\\"}"}. '
                        "Read tool errors and correct the argument keys. "
                        "Available tool schemas: "
                        + json.dumps([convert_to_openai_tool(t) for t in TOOLS])
                        + " "
                        "The stores and policies are local demo fixtures, not live market data."
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
                    "schema": ActionDecision.model_json_schema(),
                },
            },
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        content = response.choices[0].message.content or ""
        try:
            decision = parse_action(content)
        except ValueError as error:
            debug_dir = Path(tempfile.mkdtemp(prefix="codex-ifm-response-", dir="/tmp"))
            (debug_dir / "response.txt").write_text(content)
            reason = response.choices[0].finish_reason
            raise RuntimeError(
                f"IFM returned no valid structured action (finish_reason={reason}, length={len(content)}, response={debug_dir / 'response.txt'})"
            ) from error
        if (decision.action == "tools") != bool(decision.calls):
            raise ValueError(
                "Tool decisions require calls; finish decisions require an empty calls list"
            )
        calls = [
            {
                "name": call.name,
                "args": json.loads(call.arguments_json),
                "id": "call_" + uuid4().hex[:16],
                "type": "tool_call",
            }
            for call in decision.calls
        ]
        if len(calls) > 8:
            raise ValueError("At most eight tool calls are allowed per model response")
        message = AIMessage(content=decision.answer, tool_calls=calls)
        print(
            f"LLM call {state['steps'] + 1}: {[call['name'] for call in calls] or 'final answer'}",
            flush=True,
        )
        return {"messages": [message], "steps": state["steps"] + 1}

    async def review(state):
        observations = []
        for message in reversed(state["messages"]):
            if message.type != "tool":
                break
            observations.append(f"{message.name}: {message.content}")
        report = next(
            (
                message.content
                for message in reversed(state["messages"])
                if message.type == "tool"
                and message.name == "format_report"
                and getattr(message, "status", "success") != "error"
            ),
            "",
        )
        return {
            "report": report,
            "activity": "Reviewed "
            + str(len(observations))
            + " actual tool observations",
        }

    async def verify(state):
        if state.get("report"):
            return {
                "messages": [AIMessage(content=state["report"])],
                "activity": "Final answer: " + state["report"],
            }
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls or not last.content:
            raise ValueError("The agent has not produced a final answer")
        return {"activity": "Final answer: " + str(last.content)}

    builder = StateGraph(State)

    def add(name, func, role, label, description):
        builder.add_node(
            name,
            func,
            metadata={
                "langgraphics_role": role,
                "langgraphics_label": label,
                "langgraphics_description": description,
            },
        )

    add(
        "prepare",
        prepare,
        "executor",
        "Validate request",
        "Checks that an actual user message is present",
    )
    add(
        "planner",
        planner,
        "planner",
        "Plan tool use",
        "IFM LLM creates an execution plan from the request",
    )
    add(
        "agent",
        agent,
        "llm",
        "Choose actions",
        "IFM LLM selects structured actions; LangGraph executes the real tool calls",
    )
    add(
        "tools",
        ToolNode(TOOLS, handle_tool_errors=True),
        "executor",
        "Tool executor",
        "Runs only requested tools; independent calls execute concurrently",
    )
    add(
        "review",
        review,
        "executor",
        "Review observations",
        "Counts actual tool responses and returns control to the LLM",
    )
    add(
        "verify",
        verify,
        "executor",
        "Validate answer",
        "Returns the actual formatted tool report or checks the final message",
    )
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else "verify",
        {"tools": "tools", "verify": "verify"},
    )
    builder.add_edge("tools", "review")
    builder.add_conditional_edges(
        "review",
        lambda state: "verify" if state.get("report") else "agent",
        {"verify": "verify", "agent": "agent"},
    )
    builder.add_edge("verify", END)
    return builder.compile()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("../practice_agent/.env"))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--ws-port", type=int, default=8775)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--trace-file", type=Path, help="Save the actual trace locally for later replay"
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)
    key = os.getenv("IFM_API_KEY")
    if not key:
        parser.error("IFM_API_KEY is missing")
    model = os.getenv("IFM_MODEL", "IFM/K2-Horizon-375B-A23B")
    async with AsyncOpenAI(
        api_key=key, base_url="https://api.ifm.ai/v1", max_retries=0, timeout=90
    ) as client:
        viewer = watch(
            build_agent(client, model),
            port=args.port,
            ws_port=args.ws_port,
            open_browser=not args.no_open_browser,
            theme="dark",
        )
        print(
            f"Model: {model}\nViewer: http://localhost:{args.port}?ws_port={args.ws_port}&theme=dark",
            flush=True,
        )
        try:
            result = await viewer.ainvoke(
                {
                    "messages": [HumanMessage(content=args.question)],
                    "steps": 0,
                    "activity": "",
                },
                config={"recursion_limit": 60, "max_concurrency": 4},
            )
            print(result["messages"][-1].content, flush=True)
            if args.trace_file:
                viewer.export_trace(args.trace_file)
                print(f"Recorded trace: {args.trace_file}", flush=True)
            print("Run complete; viewer stays open until Ctrl+C.", flush=True)
            await asyncio.Event().wait()
        except Exception as error:  # noqa: BLE001 - keep the failed trace available for inspection
            if args.trace_file:
                viewer.export_trace(args.trace_file)
            print(
                f"Run failed: {type(error).__name__}: {error}. Viewer stays open for inspection.",
                flush=True,
            )
            await asyncio.Event().wait()
        finally:
            await viewer.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
