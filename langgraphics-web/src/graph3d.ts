import dagre from '@dagrejs/dagre';
import type {Edge, Node} from '@xyflow/react';
import type {EdgeData, NodeData} from './types';

export interface Point3 {x: number; y: number; z: number}

// Resolve nested React Flow coordinates before adding a depth axis. Stable positions
// are independent of execution status so the camera doesn't jump at every step.
export function positions3D(nodes: Node<NodeData>[], edges: Edge<EdgeData>[] = [], direction: "TB" | "LR" = "TB"): Map<string, Point3> {
    const positions = new Map<string, Point3>();
    const roots = nodes.filter(node => !node.parentId);
    const rootIds = new Set(roots.map(node => node.id));
    const graph = new dagre.graphlib.Graph();
    graph.setDefaultEdgeLabel(() => ({}));
    graph.setGraph({rankdir: direction, ranksep: 160, nodesep: 180});
    for (const node of roots) graph.setNode(node.id,
        node.data.nodeType === 'subgraph' ? {width: 950, height: 750} : {width: 270, height: 100});
    for (const edge of edges) if (rootIds.has(edge.source) && rootIds.has(edge.target)) graph.setEdge(edge.source, edge.target);
    dagre.layout(graph);
    for (const node of roots) {
        const position = graph.node(node.id);
        positions.set(node.id, {x: position.x, y: -position.y, z: 0});
    }
    // Nested scopes occupy real depth rather than squeezing all tools onto a
    // single 2D rank. IDs and clusters come from the actual compiled topology.
    const pending = nodes.filter(node => node.parentId);
    for (let pass = 0; pass < nodes.length && pending.length; pass++) {
        for (let index = pending.length - 1; index >= 0; index--) {
            const node = pending[index];
            const parent = positions.get(node.parentId!);
            if (!parent) continue;
            const siblings = nodes.filter(other => other.parentId === node.parentId);
            const tools = siblings.filter(other => other.data.nodeType !== 'start' && other.data.nodeType !== 'end');
            const toolIndex = tools.findIndex(other => other.id === node.id);
            let offset: Point3;
            if (node.data.nodeType === 'start') offset = {x: -80, y: 90, z: 35};
            else if (node.data.nodeType === 'end') offset = {x: 80, y: -90, z: -35};
            else {
                const angle = toolIndex * Math.PI * 2 / Math.max(tools.length, 1);
                const spread = Math.max(240, 105 * Math.sqrt(tools.length));
                offset = {x: Math.cos(angle) * spread * 1.8, y: Math.sin(angle) * spread * 1.1,
                    z: Math.sin(angle * 2) * spread * .6};
            }
            positions.set(node.id, {x: parent.x + offset.x, y: parent.y + offset.y, z: parent.z + offset.z});
            pending.splice(index, 1);
        }
    }
    // Unknown/malformed parents still get a finite position.
    for (const node of pending) positions.set(node.id, {x: node.position.x, y: -node.position.y, z: 0});
    if (positions.size) {
        const values = [...positions.values()];
        const center = {x: 0, y: 0, z: 0};
        for (const axis of ['x', 'y', 'z'] as const) {
            center[axis] = (Math.min(...values.map(p => p[axis])) + Math.max(...values.map(p => p[axis]))) / 2;
        }
        for (const point of values) {point.x -= center.x; point.y -= center.y; point.z -= center.z;}
    }
    return positions;
}

export function pathNodes(edges: Edge<EdgeData>[]): Set<string> {
    const ids = new Set<string>();
    for (const edge of edges) {
        if (edge.data?.status && edge.data.status !== 'idle') {ids.add(edge.source); ids.add(edge.target);}
    }
    return ids;
}
