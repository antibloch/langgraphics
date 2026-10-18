"""Per-graph observer for LangGraph 1.2 compiled StateGraphs.

Observe the executable task's scheduling writes, forwarding the exact same writes
unchanged. This adapter is intentionally isolated: the Pregel send hook is private.
No global patch, graph mutation, replacement router, or second execution is used.
"""

from copy import copy
from importlib.metadata import version
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph.state import CompiledStateGraph

SEND_KEY = "__pregel_send"
TASK_KEY = "__pregel_task_id"


def prompt_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return value if isinstance(value, str) else ""
    for key in ("question", "problem", "prompt", "input", "query", "text"):
        if isinstance(value.get(key), str):
            return value[key]
    messages = value.get("messages", [])
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        elif isinstance(message, (tuple, list)) and len(message) == 2:
            role, content = message
        else:
            role, content = (
                getattr(message, "type", None),
                getattr(message, "content", None),
            )
        if role in ("user", "human") and isinstance(content, str):
            return content
    return ""


class ObservedTask(Runnable):
    def __init__(self, original, observer, name, prefix, triggers):
        self.original, self.observer = original, observer
        self.name, self.prefix, self.triggers = name, prefix, triggers

    def _prepare(self, input, config):
        cfg = {**(config or {})}
        configurable = {**cfg.get("configurable", {})}
        if SEND_KEY not in configurable or TASK_KEY not in configurable:
            raise RuntimeError(
                "Langgraphics requires the LangGraph 1.2 task send hook; refusing to infer routes"
            )
        original_send = configurable[SEND_KEY]
        writes = []

        def observe(batch):
            batch = list(batch)
            original_send(batch)
            writes.extend(batch)

        configurable[SEND_KEY] = observe
        cfg["configurable"] = configurable
        context = self.observer.task_start(self.name, self.prefix, input, cfg)
        return cfg, writes, context

    def invoke(self, input, config=None, **kwargs):
        cfg, writes, context = self._prepare(input, config)
        try:
            result = self.original.invoke(input, cfg, **kwargs)
        except BaseException as error:
            self.observer.task_error(context, error)
            raise
        self.observer.task_end(context, input, result, writes, self.triggers)
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        cfg, writes, context = self._prepare(input, config)
        try:
            result = await self.original.ainvoke(input, cfg, **kwargs)
        except BaseException as error:
            self.observer.task_error(context, error)
            raise
        self.observer.task_end(context, input, result, writes, self.triggers)
        return result


class RecoveryWriter(Runnable):
    """Observe the engine's ParentCommand recovery path without changing writers."""

    def __init__(self, original, observer, triggers):
        self.original, self.observer, self.triggers = original, observer, triggers

    def invoke(self, input, config=None, **kwargs):
        cfg = {**config}
        configurable = {**cfg["configurable"]}
        original_send = configurable[SEND_KEY]
        writes = []

        def observe(batch):
            batch = list(batch)
            original_send(batch)
            writes.extend(batch)

        configurable[SEND_KEY] = observe
        cfg["configurable"] = configurable
        result = self.original.invoke(input, cfg, **kwargs)
        context = self.observer.contexts[configurable[TASK_KEY]]
        self.observer.task_routes(
            context, input, result, writes, self.triggers, terminate=False
        )
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)


def validate_supported(graph):
    if tuple(version("langgraph").split(".")[:2]) != ("1", "2"):
        raise RuntimeError(
            "This scheduler observer supports LangGraph 1.2.x; install a supported version before tracing"
        )
    if not isinstance(graph, CompiledStateGraph):
        raise TypeError(
            "watch currently supports compiled LangGraph StateGraph instances"
        )
    if graph.cache is not None:
        raise NotImplementedError(
            "Cache-hit provenance is not yet supported; this observer will not bypass your cache or draw inferred cache routes"
        )
    for node in graph.nodes.values():
        if isinstance(node.bound, CompiledStateGraph):
            validate_supported(node.bound)


def observe_graph(graph, observer, prefix=""):
    if not prefix:
        validate_supported(graph)
    observed = copy(graph)
    observed.nodes = {}
    triggers = {name: tuple(node.triggers) for name, node in graph.nodes.items()}
    for name, original in graph.nodes.items():
        node = original
        if isinstance(original.bound, CompiledStateGraph):
            child_prefix = f"{prefix}:{name}" if prefix else name
            child = observe_graph(original.bound, observer, child_prefix)
            node = original.copy({"bound": child, "subgraphs": [child]})
        executable = node.node
        if executable is not None:
            # Keep the engine's writer list intact: ParentCommand is handled
            # by the engine invoking these writers outside the task runnable.
            normal = node.copy({})
            normal.writers = [
                RecoveryWriter(writer, observer, triggers)
                for writer in node.flat_writers
            ]
            normal.__dict__["node"] = ObservedTask(
                executable, observer, name, prefix, triggers
            )
            node = normal
        observed.nodes[name] = node
    return observed
