import asyncio
import json
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt

from langgraphics import watch
from tests.lib.conftest import find_free_port


def viewer(graph):
    return watch(
        graph, port=find_free_port(), ws_port=find_free_port(), open_browser=False
    )


def events(view):
    return [json.loads(raw) for raw in view.ws.replay]


def pairs(view):
    return [
        (e["source"], e["target"]) for e in events(view) if e["type"] == "edge_active"
    ]


class State(TypedDict):
    values: Annotated[list[int], add]
    count: int


async def test_command_and_send_routes_are_observed_without_declared_path_map():
    def dispatch(state):
        return Command(
            goto=[Send("worker", {"count": n, "values": []}) for n in (2, 3, 4)]
        )

    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("worker", lambda s: {"values": [s["count"] ** 2]})
    graph.add_node("collect", lambda s: {"count": sum(s["values"])})
    graph.add_edge(START, "dispatch")
    graph.add_edge("worker", "collect")
    graph.add_edge("collect", END)
    plain = graph.compile()
    viewed = viewer(plain)
    try:
        assert await viewed.ainvoke({"values": [], "count": 0}) == await plain.ainvoke(
            {"values": [], "count": 0}
        )
        assert pairs(viewed).count(("dispatch", "worker")) == 3
        assert pairs(viewed).count(("worker", "collect")) == 3
        assert ("collect", END) in pairs(viewed)
        calls = [
            e
            for e in events(viewed)
            if e["type"] == "node_start" and e["node"] == "worker"
        ]
        assert len({call["task_id"] for call in calls}) == 3
        assert [
            json.loads(e["payload"])["count"]
            for e in events(viewed)
            if e["type"] == "edge_active" and e.get("kind") == "send"
        ] == [2, 3, 4]
        assert any(
            e["type"] == "graph_edge"
            and e["source"] == "dispatch"
            and e["target"] == "worker"
            for e in events(viewed)
        )
        assert plain.nodes["dispatch"].bound.__class__.__name__ != "ObservedTask"
    finally:
        await viewed.shutdown()


async def test_join_and_parallel_work_have_real_task_lifetimes():
    async def left(state):
        await asyncio.sleep(0.01)
        return {"values": [2]}

    async def right(state):
        await asyncio.sleep(0.02)
        return {"values": [3]}

    builder = StateGraph(State)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("join", lambda s: {"count": sum(s["values"])})
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge(["left", "right"], "join")
    builder.add_edge("join", END)
    view = viewer(builder.compile())
    try:
        assert (await view.ainvoke({"values": [], "count": 0}))["count"] == 5
        trace = events(view)
        join_time = next(
            e["elapsed_ms"]
            for e in trace
            if e["type"] == "node_start" and e["node"] == "join"
        )
        ends = [
            e["elapsed_ms"]
            for e in trace
            if e["type"] == "node_output" and e["node_id"] in ("left", "right")
        ]
        assert join_time >= max(ends)
        assert set(pairs(view)) == {
            (START, "left"),
            (START, "right"),
            ("left", "join"),
            ("right", "join"),
            ("join", END),
        }
    finally:
        await view.shutdown()


async def test_static_subgraph_is_observed_in_its_actual_namespace():
    child = (
        StateGraph(State)
        .add_node("work", lambda s: {"count": s["count"] + 1})
        .add_edge(START, "work")
        .add_edge("work", END)
        .compile()
    )
    outer = (
        StateGraph(State)
        .add_node("nested", child)
        .add_edge(START, "nested")
        .add_edge("nested", END)
        .compile()
    )
    view = viewer(outer)
    try:
        assert await view.ainvoke({"values": [], "count": 1}) == await outer.ainvoke(
            {"values": [], "count": 1}
        )
        assert ("nested:__start__", "nested:work") in pairs(view)
        assert ("nested:work", "nested:__end__") in pairs(view)
        start = next(
            e
            for e in events(view)
            if e["type"] == "node_start" and e["node"] == "nested:work"
        )
        assert "nested:" in start["namespace"] and "|work:" in start["namespace"]
    finally:
        await view.shutdown()


async def test_stream_shapes_values_updates_lists_and_versions_are_unchanged():
    plain = (
        StateGraph(State)
        .add_node("a", lambda s: {"count": s["count"] + 1})
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile()
    )
    view = viewer(plain)
    try:
        for mode in ("values", "updates", ["updates", "values"]):
            for version in ("v1", "v2"):
                kwargs = {"stream_mode": mode, "version": version}
                baseline = [
                    x async for x in plain.astream({"values": [], "count": 0}, **kwargs)
                ]
                actual = [
                    x async for x in view.astream({"values": [], "count": 0}, **kwargs)
                ]
                assert actual == baseline
        assert view.invoke({"values": [], "count": 0}) == plain.invoke(
            {"values": [], "count": 0}
        )
        assert list(
            view.stream({"values": [], "count": 0}, stream_mode="updates")
        ) == list(plain.stream({"values": [], "count": 0}, stream_mode="updates"))
        assert await view.ainvoke(
            {"values": [], "count": 0}, stream_mode="updates"
        ) == await plain.ainvoke({"values": [], "count": 0}, stream_mode="updates")
    finally:
        await view.shutdown()


async def test_messages_prompt_interrupt_resume_and_checkpoint_are_preserved():
    def approval(state):
        answer = interrupt("Approve?")
        return {"messages": [("assistant", str(answer))]}

    graph = (
        StateGraph(MessagesState)
        .add_node("approval", approval)
        .add_edge(START, "approval")
        .add_edge("approval", END)
        .compile(checkpointer=InMemorySaver())
    )
    view = viewer(graph)
    config = {"configurable": {"thread_id": "observed"}}
    try:
        paused = await view.ainvoke(
            {"messages": [HumanMessage(content="Original prompt")]}, config
        )
        assert "__interrupt__" in paused
        assert events(view)[0]["prompt"] == "Original prompt"
        assert events(view)[-1]["status"] == "interrupted"
        resumed = await view.ainvoke(Command(resume="Approved"), config)
        assert resumed["messages"][-1].content == "Approved"
        assert (await view.aget_state(config)).values == (
            await graph.aget_state(config)
        ).values
    finally:
        await view.shutdown()


async def test_retry_does_not_report_a_failed_run_or_emit_uncommitted_routes():
    calls = []

    def flaky(state):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("retry me")
        return {"count": 7}

    graph = (
        StateGraph(State)
        .add_node(
            "flaky",
            flaky,
            retry_policy=RetryPolicy(
                max_attempts=2, initial_interval=0.001, retry_on=ValueError
            ),
        )
        .add_edge(START, "flaky")
        .add_edge("flaky", END)
        .compile()
    )
    view = viewer(graph)
    try:
        assert (await view.ainvoke({"values": [], "count": 0}))["count"] == 7
        assert len(calls) == 2
        assert not any(e["type"] == "error" for e in events(view))
        assert pairs(view).count(("flaky", END)) == 1
    finally:
        await view.shutdown()


async def test_event_stream_v3_uses_the_same_observer_and_preserves_output():
    graph = (
        StateGraph(State)
        .add_node("a", lambda s: {"count": s["count"] + 1})
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile()
    )
    view = viewer(graph)
    try:
        run = await view.astream_events({"values": [], "count": 8}, version="v3")
        assert await run.output() == {"values": [], "count": 9}
        assert pairs(view) == [(START, "a"), ("a", END)]
        assert events(view)[-1]["type"] == "run_end"
        assert not view.busy
        sync_run = view.stream_events({"values": [], "count": 9}, version="v3")
        assert sync_run.output == {"values": [], "count": 10}
        assert not view.busy
    finally:
        await view.shutdown()


async def test_subgraph_command_return_to_parent_is_control_flow_not_failure():
    def handoff(state):
        return Command(graph=Command.PARENT, goto="after", update={"count": 9})

    child = (
        StateGraph(State)
        .add_node("handoff", handoff)
        .add_edge(START, "handoff")
        .compile()
    )
    outer = (
        StateGraph(State)
        .add_node("child", child, destinations=("after",))
        .add_node("after", lambda s: {"count": s["count"] + 1})
        .add_edge(START, "child")
        .add_edge("after", END)
        .compile()
    )
    view = viewer(outer)
    try:
        assert await view.ainvoke({"values": [], "count": 0}) == await outer.ainvoke(
            {"values": [], "count": 0}
        )
        assert not any(e["type"] == "error" for e in events(view))
        assert not any(
            e.get("node_id") == "child:handoff" and e.get("status") == "error"
            for e in events(view)
        )
        assert ("child", "after") in pairs(view)
    finally:
        await view.shutdown()
