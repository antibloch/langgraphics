import {Position} from "@xyflow/react";

export type ViewMode = "auto" | "manual";
export type InspectorMode = "off" | "tree" | "full";

export interface NodeMetrics {
    latency: string;
    costs: { cached: string; total: string };
    tokens: { cached: number; total: number };
}

export type NodeStatus = "idle" | "active" | "completed" | "error";
export type EdgeStatus = "idle" | "active" | "traversed" | "error";

export interface NodeHandle {
    id: string;
    position: Position;
    type: "source" | "target";
    style: { top?: string; left?: string; transform: string };
}

export type NodeKind = "llm" | "chain" | "tool" | "retriever" | "embedding" | "prompt" | "parser" | "chat_model";

export type NodeRole = 'input' | 'output' | 'llm' | 'planner' | 'executor' | 'tool' | 'memory' | 'router' | 'chain';
export interface NodePresentation {role?: NodeRole; description?: string; label?: string}
export interface EventTiming {seq?: number; elapsed_ms?: number}

export interface NodeData extends Record<string, unknown> {
    label: string;
    status: NodeStatus;
    handles: NodeHandle[];
    role?: NodeRole;
    description?: string;
    nodeType: "start" | "end" | "node" | "subgraph";
}

export interface EdgeData extends Record<string, unknown> {
    conditional: boolean;
    label: string | null;
    status: EdgeStatus;
}

export interface SubgraphTopology {
    nodes: ProtocolNode[];
    edges: ProtocolEdge[];
}

export interface ProtocolNode extends NodePresentation {
    id: string;
    name: string;
    node_type: "start" | "end" | "node" | "subgraph";
    subgraph?: SubgraphTopology;
}

export interface ProtocolEdge {
    id: string;
    source: string;
    target: string;
    conditional: boolean;
    label: string | null;
}

export interface GraphMessage {
    type: "graph";
    nodes: ProtocolNode[];
    edges: ProtocolEdge[];
}

export interface RunStartMessage extends EventTiming {
    type: "run_start";
    prompt?: string;
    trace_source?: string;
    run_id: string;
}

export interface RunEndMessage extends EventTiming {
    type: "run_end";
    output_summary?: string;
    status?: string;
    run_id: string;
}

export interface NodeStartMessage extends EventTiming {
    type: "node_start";
    summary?: string;
    input?: string;
    input_kind?: "prompt_context" | "state" | "tool_arguments";
    node: string;
    task_id: string;
}

export interface NodeEndMessage extends EventTiming {
    type: "node_end";
    node: string;
    task_id: string;
}

export interface EdgeActiveMessage extends EventTiming {
    type: "edge_active";
    source: string;
    target: string;
    edge_id: string;
    kind?: string;
    source_task_id?: string;
    payload?: string;
}

export interface ErrorMessage extends EventTiming {
    type: "error";
    source: string;
    target: string;
    edge_id: string | null;
}

export interface NodeMessage extends EventTiming {
    type: "node_output";
    summary?: string;
    run_id: string;
    node_id: string;
    node_kind?: NodeKind | null;
    parent_run_id?: string | null;
    status?: "ok" | "error";
    input?: string | null;
    output?: string | null;
    state?: string | null;
    metrics?: NodeMetrics | null;
}

export type NodeEntry = Omit<NodeMessage, "type">;

export interface GraphEdgeMessage extends ProtocolEdge {type: "graph_edge"}
export interface RunInterruptMessage extends EventTiming {type: "run_interrupt"; node: string; task_id: string}

export type WsMessage =
    | GraphEdgeMessage | RunInterruptMessage | GraphMessage | RunStartMessage | RunEndMessage | NodeStartMessage
    | NodeEndMessage | EdgeActiveMessage | ErrorMessage | NodeMessage;

export type ExecutionEvent =
    | RunInterruptMessage | RunStartMessage | RunEndMessage | NodeStartMessage | NodeEndMessage
    | EdgeActiveMessage | ErrorMessage | NodeMessage;
