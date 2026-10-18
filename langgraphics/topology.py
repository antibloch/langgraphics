from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode


def presentation(node_id, node):
    metadata = node.metadata or {}
    role = metadata.get("langgraphics_role")
    if role not in {"llm", "planner", "executor", "tool", "memory", "router", "chain"}:
        role = "tool" if isinstance(node.data, ToolNode) else "chain"
    if isinstance(node.data, ToolNode):
        role = "executor"
    role = {"__start__": "input", "__end__": "output"}.get(node_id, role)
    return {
        "role": role,
        "label": metadata.get(
            "langgraphics_label",
            {"__start__": "User prompt", "__end__": "Final answer"}.get(
                node_id,
                "Tool executor" if isinstance(node.data, ToolNode) else node.name,
            ),
        ),
        "description": metadata.get("langgraphics_description", ""),
    }


def tool_topology(tool_node):
    tools = list(tool_node.tools_by_name.values())
    nodes = [
        {
            "id": "__start__",
            "name": "Tool call",
            "label": "Tool call",
            "role": "executor",
            "node_type": "start",
        },
        *[
            {
                "id": f"tool:{tool.name}",
                "name": tool.name,
                "label": tool.name,
                "role": "tool",
                "node_type": "node",
                "description": tool.description,
            }
            for tool in tools
        ],
        {
            "id": "__end__",
            "name": "Tool output",
            "label": "Tool output",
            "role": "executor",
            "node_type": "end",
        },
    ]
    edges = []
    for i, tool in enumerate(tools):
        edges.extend(
            [
                {
                    "id": f"call-{i}",
                    "source": "__start__",
                    "target": f"tool:{tool.name}",
                    "conditional": True,
                    "label": "Tool call",
                },
                {
                    "id": f"result-{i}",
                    "source": f"tool:{tool.name}",
                    "target": "__end__",
                    "conditional": False,
                    "label": "Tool output",
                },
            ]
        )
    return {"nodes": nodes, "edges": edges}


def extract(graph: Any) -> dict[str, Any]:
    raw = graph.get_graph()

    return {
        "type": "graph",
        "nodes": [
            {
                "id": node_id,
                "name": node.name,
                **presentation(node_id, node),
                "node_type": {
                    "__end__": "end",
                    "__start__": "start",
                }.get(node.name, "node"),
                **(
                    {
                        "node_type": "subgraph",
                        "subgraph": extract(node.data),
                    }
                    if isinstance(node.data, CompiledStateGraph)
                    else {"node_type": "subgraph", "subgraph": tool_topology(node.data)}
                    if isinstance(node.data, ToolNode)
                    else {}
                ),
            }
            for node_id, node in raw.nodes.items()
        ],
        "edges": [
            {
                "id": f"e{i}",
                "source": edge.source,
                "target": edge.target,
                "conditional": edge.conditional,
                "label": str(edge.data) if edge.data is not None else None,
            }
            for i, edge in enumerate(raw.edges)
        ],
    }
