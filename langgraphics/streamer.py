import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from langchain_core.tracers.base import AsyncBaseTracer
from langgraph.errors import GraphBubbleUp, GraphInterrupt

from .runtime import observe_graph, prompt_text


def json_text(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda obj: (
            obj.model_dump() if hasattr(obj, "model_dump") else str(obj)
        ),
    )


def short_output(output):
    if isinstance(output, (str, int, float, bool)):
        return str(output)
    if isinstance(output, dict):
        if isinstance(output.get("activity"), str):
            return output["activity"]
        for key in ("answer", "response", "result"):
            if isinstance(output.get(key), (str, int, float)):
                return str(output[key])
        messages = output.get("messages", [])
        if messages:
            last = messages[-1]
            calls = getattr(last, "tool_calls", None)
            if calls:
                return "Tool calls: " + "; ".join(
                    f"{c['name']}({json_text(c.get('args', {}))})" for c in calls
                )
            content = getattr(last, "content", "")
            if isinstance(content, str):
                return content
    return ""


class BroadcastingTracer(AsyncBaseTracer):
    """Native tool-call lifecycle, correlated to the containing runtime task."""

    def __init__(self, viewport):
        super().__init__(_schema_format="original+chat")
        self.viewport = viewport
        self.tools = {}
        self.raw_tool_inputs = {}

    async def _persist_run(self, run):
        pass

    async def _on_chain_end(self, run):
        if self.viewport.event_run_id and run.parent_run_id is None:
            self.viewport._finish(self.viewport.event_run_id, run.outputs)
            self.viewport.event_run_id = None
            self.viewport.busy = False

    async def _on_chain_error(self, run):
        if self.viewport.event_run_id and run.parent_run_id is None:
            self.viewport._failure(RuntimeError(str(run.error)))
            self.viewport.event_run_id = None
            self.viewport.busy = False

    def _owner(self, run):
        current = run
        while current:
            metadata = (current.extra or {}).get("metadata", {})
            ns = metadata.get("langgraph_checkpoint_ns")
            if ns:
                names = [part.rsplit(":", 1)[0] for part in ns.split("|")]
                return ":".join(names)
            current = (
                self.run_map.get(str(current.parent_run_id))
                if current.parent_run_id
                else None
            )
        return None

    async def on_tool_start(
        self, serialized, input_str, *, run_id, inputs=None, **kwargs
    ):
        # LangChain may filter invalid arguments out of `inputs` before tool
        # validation. Preserve the actual call string as well, including errors.
        self.raw_tool_inputs[str(run_id)] = inputs if inputs else input_str
        await super().on_tool_start(
            serialized, input_str, run_id=run_id, inputs=inputs, **kwargs
        )

    async def _on_tool_start(self, run):
        owner = self._owner(run)
        node_id = f"{owner}:tool:{run.name}"
        if node_id not in self.viewport.node_ids:
            return
        invocation = str(run.id)
        self.tools[invocation] = (owner, node_id)
        metadata = (run.extra or {}).get("metadata", {})
        actual_input = self.raw_tool_inputs.get(invocation, run.inputs)
        self.viewport.publish(
            {
                "type": "edge_active",
                "source": f"{owner}:__start__",
                "target": node_id,
                "edge_id": self.viewport.edge_id(f"{owner}:__start__", node_id),
                "invocation_id": invocation,
                "tool_call_id": metadata.get("tool_call_id"),
                "payload": json_text(actual_input),
                "kind": "tool_call",
            }
        )
        self.viewport.publish(
            {
                "type": "node_start",
                "node": node_id,
                "task_id": invocation,
                "summary": f"{run.name}({json_text(actual_input)})",
                "input_kind": "tool_arguments",
            }
        )

    async def _on_tool_end(self, run):
        self._tool_done(run)

    async def _on_tool_error(self, run):
        self._tool_done(run)

    def _tool_done(self, run):
        identity = self.tools.pop(str(run.id), None)
        if identity is None:
            return
        owner, node_id = identity
        output = (run.outputs or {}).get("output", run.outputs)
        content = getattr(output, "content", output)
        text = (
            str(run.error)
            if run.error
            else (content if isinstance(content, str) else json_text(content))
        )
        actual_input = self.raw_tool_inputs.pop(str(run.id), run.inputs)
        self.viewport.publish(
            {
                "type": "node_output",
                "node_id": node_id,
                "run_id": str(run.id),
                "status": "error" if run.error else "ok",
                "summary": text,
                "input": json_text(actual_input),
                "output": text,
                "node_kind": "tool",
            }
        )
        if not run.error:
            self.viewport.publish(
                {
                    "type": "edge_active",
                    "source": node_id,
                    "target": f"{owner}:__end__",
                    "edge_id": self.viewport.edge_id(node_id, f"{owner}:__end__"),
                    "invocation_id": str(run.id),
                    "payload": text,
                    "kind": "observation",
                }
            )


class Viewport:
    def __init__(self, graph, ws, edge_lookup, http_server):
        self.ws, self.graph, self.edge_lookup, self.http_server = (
            ws,
            graph,
            edge_lookup,
            http_server,
        )
        self.lock = threading.RLock()
        self.started_at, self.sequence = time.monotonic(), 0
        self.active = {}
        self.contexts = {}
        self.arrivals = {}
        self.last_inputs = {}
        self.terminal_sources = set()
        self.node_ids = set()
        self.dynamic_edges = []
        self.busy = False
        self.event_run_id = None
        self.last_failure = None
        self.interrupted = False
        self.run_ended = False
        self.last_root_result = None
        self.current_prompt = ""

        def collect(topology, prefix=""):
            for node in topology["nodes"]:
                node_id = f"{prefix}:{node['id']}" if prefix else node["id"]
                self.node_ids.add(node_id)
                if node.get("subgraph"):
                    collect(node["subgraph"], node_id)
            for edge in topology["edges"]:
                if edge["target"] == "__end__":
                    source = f"{prefix}:{edge['source']}" if prefix else edge["source"]
                    self.terminal_sources.add(source)

        collect(json.loads(ws.topology_json))
        self.observed = observe_graph(graph, self)

    def __getattr__(self, name):
        if name in {
            "batch",
            "abatch",
            "batch_as_completed",
            "abatch_as_completed",
            "with_config",
            "with_retry",
            "with_fallbacks",
            "astream_log",
        }:
            raise NotImplementedError(
                f"Apply {name} to the compiled graph before watch(); this viewport supports one traced run at a time"
            )
        return getattr(self.graph, name)

    def publish(self, message):
        with self.lock:
            self.sequence += 1
            message = {
                **message,
                "seq": self.sequence,
                "elapsed_ms": round((time.monotonic() - self.started_at) * 1000, 3),
            }
            encoded = json_text(message)
            self.ws.record(encoded)
            if self.ws.loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self.ws.broadcast(encoded), self.ws.loop
                )

    async def broadcast(self, message):
        self.publish(message)
        await asyncio.sleep(0)

    def edge_id(self, source, target):
        with self.lock:
            key = (source, target)
            if key not in self.edge_lookup:
                edge = {
                    "id": f"runtime-{uuid.uuid4().hex[:12]}",
                    "source": source,
                    "target": target,
                    "conditional": True,
                    "label": "Runtime route",
                }
                self.edge_lookup[key] = edge["id"]
                self.dynamic_edges.append(edge)
                self.publish({"type": "graph_edge", **edge})
            return self.edge_lookup[key]

    def task_start(self, name, prefix, input, config):
        metadata = config.get("metadata", {})
        task_id = config["configurable"]["__pregel_task_id"]
        node_id = f"{prefix}:{name}" if prefix else name
        context = {
            "node": node_id,
            "task_id": task_id,
            "prefix": prefix,
            "namespace": metadata.get("langgraph_checkpoint_ns", ""),
            "step": metadata.get("langgraph_step"),
            "start": time.monotonic(),
        }
        with self.lock:
            self.last_inputs[task_id] = input
            self.active[task_id] = context
            self.contexts[task_id] = context
        if name != "__start__":
            self.publish(
                {
                    "type": "node_start",
                    "node": node_id,
                    "task_id": task_id,
                    "namespace": context["namespace"],
                    "step": context["step"],
                    "input": json_text(input),
                    "input_kind": "prompt_context"
                    if self.current_prompt and prompt_text(input) == self.current_prompt
                    else "state",
                }
            )
        return context

    def task_end(self, context, input, result, writes, triggers):
        source, prefix = context["node"], context["prefix"]
        is_start = source.rsplit(":", 1)[-1] == "__start__"
        if not prefix and not is_start:
            self.last_root_result = result
        with self.lock:
            self.active.pop(context["task_id"], None)
        if not is_start:
            self.publish(
                {
                    "type": "node_output",
                    "node_id": source,
                    "run_id": context["task_id"],
                    "status": "ok",
                    "summary": short_output(result),
                    "input": json_text(input),
                    "output": json_text(result),
                    "node_kind": "chain",
                    "duration_ms": round(
                        (time.monotonic() - context["start"]) * 1000, 3
                    ),
                }
            )
        self.task_routes(context, input, result, writes, triggers)

    def task_routes(self, context, input, result, writes, triggers, terminate=True):
        source, prefix = context["node"], context["prefix"]
        is_start = source.rsplit(":", 1)[-1] == "__start__"
        destinations = []
        for channel, value in writes:
            # These are actual scheduler writes, not guesses from graph edges.
            if channel == "__pregel_tasks" and hasattr(value, "node"):
                destinations.append((value.node, "send", value.arg))
            else:
                for target, channels in triggers.items():
                    if channel in channels and target != "__start__":
                        destinations.append(
                            (
                                target,
                                "join" if channel.startswith("join:") else "state",
                                result,
                            )
                        )
        seen = context.setdefault("routes_seen", set())
        for target, kind, payload in destinations:
            # Ordinary repeated channel writes coalesce into one scheduled task;
            # Send packets remain distinct even when their destinations coincide.
            if kind != "send" and target in seen:
                continue
            seen.add(target)
            target_id = f"{prefix}:{target}" if prefix else target
            self.arrivals[target_id] = source
            self.publish(
                {
                    "type": "edge_active",
                    "source": source,
                    "target": target_id,
                    "edge_id": self.edge_id(source, target_id),
                    "kind": kind,
                    "source_task_id": context["task_id"],
                    "namespace": context["namespace"],
                    "payload": json_text(input if is_start else payload),
                }
            )
        # LangGraph 1.2 drops END writes. A successful task without any
        # outgoing scheduling writes terminates its branch; this is marked
        # separately from an observed routing write, never inferred as a call.
        if terminate and not destinations and not is_start:
            target = f"{prefix}:__end__" if prefix else "__end__"
            self.publish(
                {
                    "type": "edge_active",
                    "source": source,
                    "target": target,
                    "edge_id": self.edge_id(source, target),
                    "kind": "termination",
                    "source_task_id": context["task_id"],
                    "payload": json_text(result),
                }
            )

    def task_error(self, context, error):
        with self.lock:
            self.active.pop(context["task_id"], None)
        if isinstance(error, GraphInterrupt):
            self.interrupted = True
            self.publish(
                {
                    "type": "run_interrupt",
                    "node": context["node"],
                    "task_id": context["task_id"],
                }
            )
            return
        if isinstance(error, GraphBubbleUp):
            self.publish(
                {
                    "type": "node_output",
                    "node_id": context["node"],
                    "run_id": context["task_id"],
                    "status": "ok",
                    "summary": "Returned control to the parent graph",
                }
            )
            return
        if isinstance(
            error, (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit)
        ):
            return
        target = context["node"]
        source = self.arrivals.get(target)
        self.publish(
            {
                "type": "node_output",
                "node_id": target,
                "run_id": context["task_id"],
                "status": "error",
                "summary": str(error),
            }
        )
        self.last_failure = (source or target, target, str(error))
        self.publish(
            {
                "type": "task_attempt_error",
                "node": target,
                "task_id": context["task_id"],
                "message": str(error),
            }
        )

    def _failure(self, error):
        self.run_ended = True
        source, target, message = self.last_failure or (
            "__start__",
            "__start__",
            str(error),
        )
        self.publish(
            {
                "type": "error",
                "source": source,
                "target": target,
                "edge_id": self.edge_lookup.get((source, target)),
                "message": message,
            }
        )

    def _finish(self, run_id, result=None):
        self.run_ended = True
        if result is None:
            result = self.last_root_result
        self.publish(
            {
                "type": "run_end",
                "run_id": run_id,
                "status": "interrupted" if self.interrupted else "complete",
                "output_summary": short_output(result),
            }
        )

    def _start(self, input):
        if self.busy:
            raise RuntimeError("One run at a time per Langgraphics viewport")
        self.busy = True
        self.last_failure = None
        self.interrupted = False
        self.run_ended = False
        self.last_root_result = None
        self.started_at, self.sequence = time.monotonic(), 0
        self.active.clear()
        self.contexts.clear()
        self.arrivals.clear()
        self.last_inputs.clear()
        run_id = uuid.uuid4().hex[:8]
        self.current_prompt = prompt_text(input)
        self.publish(
            {
                "type": "run_start",
                "run_id": run_id,
                "prompt": prompt_text(input),
                "input": json_text(input),
                "trace_source": "langgraph_scheduler_writes",
            }
        )
        for edge in self.dynamic_edges:
            self.publish({"type": "graph_edge", **edge})
        return run_id

    def _config(self, config):
        merged = dict(config or {})
        callbacks = merged.get("callbacks")
        tracer = BroadcastingTracer(self)
        if callbacks is None:
            merged["callbacks"] = [tracer]
        elif isinstance(callbacks, (list, tuple)):
            merged["callbacks"] = [*callbacks, tracer]
        else:
            manager = callbacks.copy()
            manager.add_handler(tracer)
            merged["callbacks"] = manager
        return merged

    async def shutdown(self):
        await self.ws.shutdown()
        await asyncio.to_thread(self.http_server.shutdown)
        self.http_server.server_close()

    def export_trace(self, path):
        """Save actual recorded events, never re-execute the graph to export."""
        with self.lock:
            trace = {
                "format": "langgraphics.trace.v1",
                "graph": json.loads(self.ws.topology_json),
                "events": [json.loads(raw) for raw in self.ws.replay],
            }
            Path(path).write_text(json_text(trace))

    def stream_events(self, input, config=None, **kwargs):
        self.event_run_id = self._start(input)
        try:
            return self.observed.stream_events(input, self._config(config), **kwargs)
        except Exception as error:
            self._failure(error)
            self.event_run_id = None
            self.busy = False
            raise

    def astream_events(self, input, config=None, **kwargs):
        self.event_run_id = self._start(input)
        if kwargs.get("version") == "v3":

            async def create():
                try:
                    return await self.observed.astream_events(
                        input, self._config(config), **kwargs
                    )
                except Exception as error:
                    self._failure(error)
                    self.event_run_id = None
                    self.busy = False
                    raise

            return create()

        async def iterate():
            try:
                async for event in self.observed.astream_events(
                    input, self._config(config), **kwargs
                ):
                    yield event
            except Exception as error:
                self._failure(error)
                raise
            finally:
                self.event_run_id = None
                self.busy = False

        return iterate()

    async def ainvoke(self, input, config=None, **kwargs):
        run_id = self._start(input)
        try:
            result = await self.observed.ainvoke(input, self._config(config), **kwargs)
            self._finish(run_id, result)
            return result
        except Exception as error:
            self._failure(error)
            raise
        finally:
            if not self.run_ended:
                self.publish(
                    {"type": "run_end", "run_id": run_id, "status": "cancelled"}
                )
            self.busy = False

    def invoke(self, input, config=None, **kwargs):
        run_id = self._start(input)
        try:
            result = self.observed.invoke(input, self._config(config), **kwargs)
            self._finish(run_id, result)
            return result
        except Exception as error:
            self._failure(error)
            raise
        finally:
            if not self.run_ended:
                self.publish(
                    {"type": "run_end", "run_id": run_id, "status": "cancelled"}
                )
            self.busy = False

    async def astream(self, input, config=None, **kwargs) -> AsyncIterator:
        run_id = self._start(input)
        try:
            async for chunk in self.observed.astream(
                input, self._config(config), **kwargs
            ):
                yield chunk
            self._finish(run_id)
        except Exception as error:
            self._failure(error)
            raise
        finally:
            if not self.run_ended:
                self.publish(
                    {"type": "run_end", "run_id": run_id, "status": "cancelled"}
                )
            self.busy = False

    def stream(self, input, config=None, **kwargs) -> Iterator:
        run_id = self._start(input)
        try:
            yield from self.observed.stream(input, self._config(config), **kwargs)
            self._finish(run_id)
        except Exception as error:
            self._failure(error)
            raise
        finally:
            if not self.run_ended:
                self.publish(
                    {"type": "run_end", "run_id": run_id, "status": "cancelled"}
                )
            self.busy = False
