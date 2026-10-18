import {describe, expect, it} from 'vitest';
import type {Edge, Node} from '@xyflow/react';
import type {EdgeData, NodeData} from '../../langgraphics-web/src/types';
import {pathNodes, positions3D} from '../../langgraphics-web/src/graph3d';

function node(id: string, x: number, y: number, parentId?: string): Node<NodeData> {
    return {id, position: {x, y}, parentId, data: {label: id, status: 'idle', handles: [], nodeType: 'node'}};
}
describe('3D graph layout', () => {
    it('groups nested tools in finite, centered 3D scopes', () => {
        const result = positions3D([node('root', 100, 100), node('child', 20, 40, 'root'), node('last', 500, 800)]);
        expect(result.get('child')).not.toEqual(result.get('root'));
        for (const point of result.values()) Object.values(point).forEach(value => expect(Number.isFinite(value)).toBe(true));
        expect(result.size).toBe(3);
    });
    it('handles empty graphs and does not move nodes when their status changes', () => {
        expect(positions3D([]).size).toBe(0);
        const nodes = [node('a', 0, 0), node('b', 50, 100)];
        expect(positions3D(nodes)).toEqual(positions3D(nodes.map(n => ({...n, className: 'active'}))));
    });
    it('keeps error paths and excludes unvisited edges from the path filter', () => {
        const edges = [
            {id: '1', source: 'a', target: 'b', data: {status: 'traversed'}},
            {id: '2', source: 'b', target: 'c', data: {status: 'error'}},
            {id: '3', source: 'c', target: 'd', data: {status: 'idle'}},
        ] as Edge<EdgeData>[];
        expect([...pathNodes(edges)]).toEqual(['a', 'b', 'c']);
    });
});
