"""Deterministic model fixtures; tool execution itself is real LangGraph code."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

pytest.importorskip(
    "openai", reason="Install the react dependency group for example tests"
)
from examples.complex_react_ifm import DEFAULT_QUESTION, build_agent
from langgraphics import watch
from tests.lib.conftest import find_free_port


class FixtureCompletions:
    def __init__(self):
        self.index = 0
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        rounds = [
            [
                ("lookup_product", {"name": "Summit kit"}),
                ("lookup_product", {"name": "River bottle"}),
                ("search_notes", {"query": "shipping discount"}),
            ],
            [
                ("calculator", {"expression": "3*84+2*18"}),
                ("calculator", {"expression": "3*2.4+2*0.35"}),
                ("summarize_numbers", {"values": [84, 84, 84, 18, 18]}),
            ],
            [
                ("apply_discount", {"subtotal_usd": 288, "code": "WELCOME10"}),
                ("convert_units", {"value": 7.9, "from_unit": "kg", "to_unit": "g"}),
                ("shipping_quote", {"weight_kg": 7.9, "distance_km": 450}),
            ],
            [("calculator", {"expression": "259.2+21.35"})],
            [
                (
                    "format_report",
                    {
                        "title": "Verified purchase",
                        "lines": ["Delivered total: $280.55", "Mass: 7900 g"],
                    },
                )
            ],
        ]
        if self.index == 0:
            content = "Look up facts, compute totals, apply policies, and report verified observations."
        elif self.index <= len(rounds):
            content = json.dumps(
                {
                    "action": "tools",
                    "calls": [
                        {"name": name, "arguments_json": json.dumps(args)}
                        for name, args in rounds[self.index - 1]
                    ],
                    "answer": "",
                }
            )
        else:
            content = json.dumps(
                {
                    "action": "finish",
                    "calls": [],
                    "answer": "Verified purchase: Delivered total: $280.55; Mass: 7900 g",
                }
            )
        self.index += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


async def test_all_eight_tools_have_actual_invocations_payloads_and_results():
    completions = FixtureCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    graph = build_agent(client, "fixture")
    view = watch(
        graph, port=find_free_port(), ws_port=find_free_port(), open_browser=False
    )
    try:
        result = await view.ainvoke(
            {
                "messages": [HumanMessage(content=DEFAULT_QUESTION)],
                "steps": 0,
                "activity": "",
            }
        )
        assert "$280.55" in result["messages"][-1].content
        events = [json.loads(raw) for raw in view.ws.replay]
        calls = [
            e
            for e in events
            if e["type"] == "edge_active" and e.get("kind") == "tool_call"
        ]
        observations = [
            e
            for e in events
            if e["type"] == "edge_active" and e.get("kind") == "observation"
        ]
        assert len(calls) == len(observations) == 11
        names = {e["target"].split(":tool:")[-1] for e in calls}
        assert names == {
            "lookup_product",
            "search_notes",
            "calculator",
            "summarize_numbers",
            "apply_discount",
            "convert_units",
            "shipping_quote",
            "format_report",
        }
        assert {e["invocation_id"] for e in calls} == {
            e["invocation_id"] for e in observations
        }
        assert len({e["invocation_id"] for e in calls}) == 11
        assert any("Summit kit" in e["payload"] for e in calls)
        assert any("280.55" in e["payload"] for e in observations)
        assert not any(e["type"] == "error" for e in events)
        assert events[0]["prompt"] == DEFAULT_QUESTION
    finally:
        await view.shutdown()
