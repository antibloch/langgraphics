import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from langgraphics import watch
from langgraphics.topology import extract
from tests.lib.conftest import find_free_port, safe_ainvoke, ws_collect


class TraceState(TypedDict):
    question: str
    count: int
    activity: str


def loop_graph():
    async def reason(state):
        await asyncio.sleep(0.02)
        return {"count": state["count"] + 1, "activity": "Choose next action"}

    async def tool(state):
        await asyncio.sleep(0.02)
        return {"activity": "Observation: calculator result"}

    graph = StateGraph(TraceState)
    graph.add_node(
        "reason",
        reason,
        metadata={"langgraphics_role": "llm", "langgraphics_label": "Reason & choose"},
    )
    graph.add_node("calculator", tool, metadata={"langgraphics_role": "tool"})
    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        lambda s: "calculator" if s["count"] < 3 else END,
        {"calculator": "calculator", END: END},
    )
    graph.add_edge("calculator", "reason")
    return graph.compile()


def test_roles_come_from_metadata_and_unknown_nodes_stay_untyped(simple_graph):
    nodes = {node["id"]: node for node in extract(loop_graph())["nodes"]}
    assert nodes["reason"]["role"] == "llm"
    assert nodes["reason"]["label"] == "Reason & choose"
    assert nodes["calculator"]["role"] == "tool"
    assert nodes["__start__"]["role"] == "input"
    assert nodes["__end__"]["role"] == "output"
    assert (
        next(n for n in extract(simple_graph)["nodes"] if n["id"] == "step_a")["role"]
        == "chain"
    )


async def test_real_loop_events_have_exact_path_timing_prompt_and_observations():
    port = find_free_port()
    viewport = watch(
        loop_graph(), port=find_free_port(), ws_port=port, open_browser=False
    )
    async with ws_collect(port) as (messages, _done):
        await safe_ainvoke(
            viewport, {"question": "17 * 23 + 19", "count": 0, "activity": ""}
        )
    path = [(e["source"], e["target"]) for e in messages if e["type"] == "edge_active"]
    assert path == [
        (START, "reason"),
        ("reason", "calculator"),
        ("calculator", "reason"),
        ("reason", "calculator"),
        ("calculator", "reason"),
        ("reason", END),
    ]
    events = [e for e in messages if e["type"] != "graph"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["elapsed_ms"] for e in events] == sorted(e["elapsed_ms"] for e in events)
    assert events[0]["prompt"] == "17 * 23 + 19"
    assert events[-1]["elapsed_ms"] >= 100
    assert [e["node"] for e in events if e["type"] == "node_start"] == [
        "reason",
        "calculator",
        "reason",
        "calculator",
        "reason",
    ]
    assert any(e.get("summary") == "Observation: calculator result" for e in events)


async def test_unchosen_conditional_edge_does_not_animate_when_its_target_runs_later():
    graph = StateGraph(TraceState)
    for name in ("route", "first", "later"):
        graph.add_node(name, lambda state: {"count": state["count"] + 1})
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: "via_first",
        {"via_first": "first", "skip_first": "later"},
    )
    graph.add_edge("first", "later")
    graph.add_edge("later", END)
    port = find_free_port()
    viewport = watch(
        graph.compile(), port=find_free_port(), ws_port=port, open_browser=False
    )
    async with ws_collect(port) as (messages, _done):
        await safe_ainvoke(viewport, {"question": "test", "count": 0, "activity": ""})
    path = [(e["source"], e["target"]) for e in messages if e["type"] == "edge_active"]
    assert path == [
        (START, "route"),
        ("route", "first"),
        ("first", "later"),
        ("later", END),
    ]
