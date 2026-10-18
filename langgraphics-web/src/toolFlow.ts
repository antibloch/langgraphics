import type {Edge, Node} from '@xyflow/react';
import type {EdgeData, ExecutionEvent, NodeData} from './types';

/** Collapse ToolNode plumbing, preserving actual callers and continuation routes. */
export function toolFlow(nodes: Node<NodeData>[], edges: Edge<EdgeData>[], events: ExecutionEvent[]) {
    const owners = new Set(nodes.filter(node => node.data.nodeType === 'subgraph' &&
        nodes.some(child => child.parentId === node.id && child.data.role === 'tool')).map(node => node.id));
    const tools = new Map(nodes.filter(node => node.parentId && owners.has(node.parentId) && node.data.role === 'tool')
        .map(node => [node.id, node.parentId!]));
    const ports = new Set(nodes.filter(node => node.parentId && owners.has(node.parentId) && !tools.has(node.id)).map(node => node.id));
    const hidden = new Set([...owners, ...ports]);
    const resultNodes = nodes.filter(node => !hidden.has(node.id)).map(node => tools.has(node.id)
        ? {...node, parentId: nodes.find(owner => owner.id === tools.get(node.id))?.parentId} : node);
    const resultEdges = new Map(edges.filter(edge => !hidden.has(edge.source) && !hidden.has(edge.target)).map(edge => [edge.id, {...edge,data:edge.data ? {...edge.data} : undefined}]));
    const id = (kind: string, source: string, target: string) => `tool-flow:${kind}:${source}->${target}`;
    const add = (kind: string, source: string, target: string) => {
        const key = id(kind, source, target);
        if (!resultEdges.has(key)) resultEdges.set(key, {id:key,source,target,
            data:{conditional:false,label:kind === 'call' ? 'Tool call' : 'Tool output',status:'idle'}});
        return key;
    };
    // Show available tool paths faintly before execution, without a fake router actor.
    for (const [tool, owner] of tools) {
        for (const edge of edges) {
            if (edge.target === owner && !hidden.has(edge.source)) add('call',edge.source,tool);
            if (edge.source === owner && !hidden.has(edge.target)) add('output',tool,edge.target);
        }
    }
    const arrivals = new Map<string,string>();
    const projected: ExecutionEvent[] = [];
    for (const event of events) {
        if (event.type !== 'edge_active') {projected.push(event);continue;}
        if (owners.has(event.target)) {arrivals.set(event.target,event.source);continue;}
        if (event.kind === 'tool_call' && tools.has(event.target)) {
            const owner = tools.get(event.target)!;
            const caller = arrivals.get(owner);
            if (caller && !hidden.has(caller)) projected.push({...event,source:caller,
                edge_id:add('call',caller,event.target)});
        } else if (event.kind === 'observation' && tools.has(event.source)) {
            const owner = tools.get(event.source)!;
            const continuations = edges.filter(edge => edge.source === owner && !hidden.has(edge.target));
            // Output production is its own event. Never delay it until dispatch completes.
            // If routing is still ambiguous, mark production at the tool without guessing a recipient.
            const target = continuations.length === 1 ? continuations[0].target : event.source;
            projected.push({...event,target,
                edge_id:target === event.source ? event.edge_id : add('output',event.source,target)});
        } else if (owners.has(event.source)) {
            // Dispatcher plumbing stays hidden; it does not produce the tool's output.
        } else if (!hidden.has(event.source) && !hidden.has(event.target)) projected.push(event);
    }
    for (const event of projected) {
        if (event.type === 'edge_active') {
            const edge = resultEdges.get(event.edge_id);
            if (edge?.data) edge.data = {...edge.data,status:'traversed'};
        }
    }
    return {nodes:resultNodes,edges:[...resultEdges.values()],events:projected};
}
