import json

from langgraphics.broadcaster import Broadcaster


def test_completed_run_remains_available_for_late_viewers():
    broadcaster = Broadcaster({"type": "graph", "nodes": [], "edges": []})
    messages = [
        {"type": "run_start", "run_id": "one"},
        {"type": "edge_active", "source": "a", "target": "b"},
        {"type": "run_end", "run_id": "one"},
    ]
    for message in messages:
        broadcaster.record(json.dumps(message))
    assert [json.loads(raw) for raw in broadcaster.replay] == messages
    broadcaster.record(json.dumps({"type": "run_start", "run_id": "two"}))
    assert len(broadcaster.replay) == 1


def test_failed_run_preserves_path_and_error():
    broadcaster = Broadcaster({})
    for message in [{"type": "run_start"}, {"type": "edge_active"}, {"type": "error"}]:
        broadcaster.record(json.dumps(message))
    assert [json.loads(raw)["type"] for raw in broadcaster.replay] == [
        "run_start",
        "edge_active",
        "error",
    ]
