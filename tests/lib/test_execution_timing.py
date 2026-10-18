"""Compare trace intervals to independent clocks inside arbitrary graph nodes."""
import asyncio
import json
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from langgraphics import watch
from tests.lib.conftest import find_free_port


async def test_arbitrary_graph_trace_tracks_measured_parallel_execution_intervals():
    class Payload(TypedDict):
        fast: bool
        slow: bool

    measured = {}

    async def fast(_):
        measured['fast_start'] = time.monotonic()
        await asyncio.sleep(0.04)
        measured['fast_end'] = time.monotonic()
        return {'fast': True}

    async def slow(_):
        measured['slow_start'] = time.monotonic()
        await asyncio.sleep(0.14)
        measured['slow_end'] = time.monotonic()
        return {'slow': True}

    builder = StateGraph(Payload)
    builder.add_node('arbitrary_alpha', fast)
    builder.add_node('arbitrary_beta', slow)
    builder.add_node('merge', lambda state: {})
    builder.add_edge(START, 'arbitrary_alpha')
    builder.add_edge(START, 'arbitrary_beta')
    builder.add_edge(['arbitrary_alpha', 'arbitrary_beta'], 'merge')
    builder.add_edge('merge', END)
    view = watch(builder.compile(), port=find_free_port(), ws_port=find_free_port(), open_browser=False)
    try:
        assert await view.ainvoke({'fast': False, 'slow': False}) == {'fast': True, 'slow': True}
        trace = [json.loads(raw) for raw in view.ws.replay]
        intervals = {}
        for node, prefix in [('arbitrary_alpha', 'fast'), ('arbitrary_beta', 'slow')]:
            start = next(e for e in trace if e['type'] == 'node_start' and e['node'] == node)
            end = next(e for e in trace if e['type'] == 'node_output' and e['node_id'] == node)
            actual_ms = (measured[prefix + '_end'] - measured[prefix + '_start']) * 1000
            recorded_ms = end['elapsed_ms'] - start['elapsed_ms']
            # Observer serialization is outside the node's own measurement.
            assert actual_ms <= recorded_ms + 1
            assert recorded_ms - actual_ms < 50
            intervals[prefix] = (start['elapsed_ms'], end['elapsed_ms'])
        assert intervals['slow'][0] < intervals['fast'][1] < intervals['slow'][1]
        merge = next(e for e in trace if e['type'] == 'node_start' and e['node'] == 'merge')
        assert merge['elapsed_ms'] >= intervals['slow'][1]
    finally:
        await view.shutdown()
