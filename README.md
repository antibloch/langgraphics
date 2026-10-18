# Langgraphics 3D

A minimal, local 3D execution viewer for LangGraph agents. This fork of
[proactive-agent/langgraphics](https://github.com/proactive-agent/langgraphics)
replaces the inspector with a graph and a single **Pause / Resume** button.

The visual model is **agent abstraction → named tool call → tool output**.
Agent abstractions can be ReAct, planner, reflector, executor, or your own nodes.
Internal `ToolNode` dispatch plumbing is hidden. A glowing red dot appears when
a tool produces an output, without an output popup or text label.

## Install

Requires Python 3.10+, Git, and a browser with WebGL. The supported LangGraph
version is **1.2.x**, installed automatically with this fork.

```bash
python3 -m pip install "git+https://github.com/antibloch/langgraphics.git@main"
```

For a project managed with uv:

```bash
uv add "langgraphics @ git+https://github.com/antibloch/langgraphics.git@main"
```

The built frontend is included in the repository and Python package. Node.js is
only needed when changing the frontend. Installing the upstream PyPI package
with `pip install langgraphics` does not install this fork.

For local development and the examples:

```bash
git clone https://github.com/antibloch/langgraphics.git
cd langgraphics
uv sync --group react
```

For an existing project that should use your local checkout:

```bash
uv add --editable /path/to/langgraphics
```

## Use with your agent

Wrap your existing compiled graph before executing it. Keep its current model,
tools, reducers, checkpointer, store, and invocation configuration.

```python
import asyncio
from langgraphics import watch

# compiled_graph = your existing StateGraph(...).compile(...)

async def main():
    viewer = watch(compiled_graph, theme="dark")
    try:
        result = await viewer.ainvoke(
            {"messages": [("user", "Your prompt")]},
            config={"recursion_limit": 30},
        )
        print(result)
        # Keep the viewer available after execution for replay.
        await asyncio.Event().wait()
    finally:
        await viewer.shutdown()

asyncio.run(main())
```

The browser opens automatically at
`http://localhost:8764/?ws_port=8765&theme=dark`. Set `open_browser=False` to open
it yourself, or choose different ports with `port=8794, ws_port=8795`.
Drag to orbit, right-drag to pan, and scroll to zoom.

`invoke`, `ainvoke`, `stream`, `astream`, and LangGraph's event-stream interfaces
are supported. The wrapper preserves graph results and requested stream formats.
For checkpointed agents, pass your normal thread configuration to the invocation.

### Name agent abstractions

Roles are optional presentation metadata; they never change routing:

```python
builder.add_node("reason", reason, metadata={
    "langgraphics_role": "llm",
    "langgraphics_label": "ReAct agent",
})
builder.add_node("plan", plan, metadata={
    "langgraphics_role": "planner",
    "langgraphics_label": "Planner",
})
builder.add_node("reflect", reflect, metadata={
    "langgraphics_role": "executor",
    "langgraphics_label": "Reflector",
})
```

Supported roles are `llm`, `planner`, `executor`, `tool`, `memory`, `router`, and
`chain`. Unknown nodes remain workflow nodes rather than receiving guessed roles.
Registered `ToolNode` tools are recognized automatically. Only tools actually
called become active; unused tools remain grey.

## Try the three-tool ReAct agent

From the cloned repository:

```bash
export IFM_API_KEY="your-api-key"
uv run --group react python examples/three_tool_react_ifm.py
```

Alternatively, load an existing environment file without copying its credentials:

```bash
uv run --group react python examples/three_tool_react_ifm.py \
  --env-file /path/to/your/.env
```

This is a real **model → ToolNode → model** loop with exactly three tools:

- `lookup_product`: reads a local demo catalog.
- `calculator`: evaluates bounded numeric arithmetic.
- `convert_units`: converts mass or distance units.

The default prompt looks up a Summit kit, calculates the price of four kits,
and converts one kit's weight to grams. The verified result is **$336** and
**2,400 g**. The viewer opens at
`http://localhost:8794/?ws_port=8795&theme=dark`.

```bash
uv run --group react python examples/three_tool_react_ifm.py \
  --prompt "Look up the River bottle, calculate the price of 6 bottles, and convert one bottle's weight to grams." \
  --trace-file /tmp/three-tool-trace.json
```

The example calls `https://api.ifm.ai/v1`, using `IFM_MODEL` when set or
`IFM/K2-Horizon-375B-A23B` by default. It adapts structured model decisions into
standard LangChain tool calls, executes real tools through LangGraph, and feeds
actual results back to the model. It has a six-model-call budget and a 90-second
request timeout. Model requests use your provider account; the viewer itself
requires no provider key. Keys are never sent to the browser.

For a larger agent with planning, observation review, and eight tools:

```bash
uv run --group react python examples/complex_react_ifm.py \
  --env-file /path/to/your/.env
```

All catalog and shipping facts in these examples are local demonstration data.
`--no-open-browser`, `--port`, and `--ws-port` are available on both examples.
Stop their server with Ctrl+C when finished.

## Replay without a model or API key

A recorded real three-tool execution is included:

```bash
uv run python -m langgraphics.replay examples/traces/three-tool-react.json
```

For the eight-tool recording:

```bash
uv run python -m langgraphics.replay examples/traces/purchase-report.json
```

Replay opens at `http://localhost:8774/?ws_port=8775&theme=dark`. It does not
execute tools or make model requests. To save your own run:

```python
result = await viewer.ainvoke(your_input, your_config)
viewer.export_trace("/path/to/trace.json")
```

Trace exports contain actual prompts, inputs, arguments, and results. Store or
share them as you would the agent's own data. Example recordings contain only
the demonstration request and local fixture results.

## Timing and accuracy

- Runtime task starts, finishes, tool callbacks, and scheduler writes drive the
  display. Routes are not scripted or inferred from node names.
- Playback uses **1× recorded time**, including model waits. Pausing freezes the
  display only; the agent continues recording. Resume continues from that position
  at 1×, without skipping ahead or speeding up to catch up.
- Tool output glow begins at the original tool-result timestamp. Its one-second
  fade makes fast results visible; this is an afterglow, not execution or transit
  time. Node execution and the playback clock are not extended by it.
- When a run finishes, Resume starts its recorded execution again from the prompt.
- Browser frame rate, transport latency, and tracing overhead limit visual
  precision. The viewer shows execution events, not network transit or hidden
  model reasoning.

## Compatibility and limitations

The observer supports **LangGraph 1.2.x compiled `StateGraph` instances**. It uses
a private task-write hook on a per-instance graph copy, forwarding original
writes unchanged. Unsupported graph types and versions fail explicitly.

Branches, loops, joins, `Command.goto`, `Send` fan-outs, parallel tasks, retries,
interrupt/resume, checkpointers, and statically attached compiled subgraphs have
integration coverage. Functions that invoke hidden inner graphs appear as their
outer declared node. Tool output production is still marked when its continuation
route is not yet known; the viewer does not guess a recipient.

Cache-enabled graphs are currently unsupported. Configure wrappers before
calling `watch`; each viewport supports one invocation at a time. Batch execution
is unsupported. Legacy `inspect` and `mode` options remain accepted, but the old
inspector and automatic camera-follow interface are removed.

## Develop and test

```bash
uv sync --group react
uv run pytest -q
cd langgraphics-web
npm ci
npm test
npm run build
```

Rebuild after frontend changes. Commit the updated `langgraphics/static` assets
with the source so direct Git installations work without Node.js.

```bash
uv build
```

Both wheel and source distribution include the built viewer.

## License

[MIT](LICENSE). Based on LangGraphics by Artyom Vancyan and contributors.
