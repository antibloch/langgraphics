import {describe, expect, it} from 'vitest';
import {computeStatuses} from '../../langgraphics-web/src/hooks/useGraphState';
import type {ExecutionEvent} from '../../langgraphics-web/src/types';
const start: ExecutionEvent = {type:'run_start',run_id:'real',trace_source:'langgraph_scheduler_writes'};
describe('scheduler execution semantics', () => {
    it('does not mark scheduled nodes active before they actually start', () => {
        const {nodeStatuses,edgeStatuses} = computeStatuses([start,
            {type:'edge_active',source:'left',target:'join',edge_id:'e',kind:'join'}]);
        expect(nodeStatuses.get('join')).toBeUndefined();
        expect(edgeStatuses.get('e')).toBe('traversed');
    });
    it('keeps a shared node active until all concurrent Send invocations finish', () => {
        const events: ExecutionEvent[] = [start,
            {type:'node_start',node:'worker',task_id:'one'},
            {type:'node_start',node:'worker',task_id:'two'},
            {type:'node_output',node_id:'worker',run_id:'one',status:'ok'}];
        expect(computeStatuses(events).nodeStatuses.get('worker')).toBe('active');
        events.push({type:'node_output',node_id:'worker',run_id:'two',status:'ok'});
        expect(computeStatuses(events).nodeStatuses.get('worker')).toBe('completed');
    });
    it('does not paint unused tools completed when their dispatcher finishes', () => {
        const events: ExecutionEvent[] = [start,
            {type:'node_start',node:'tools',task_id:'dispatcher'},
            {type:'node_start',node:'tools:tool:calculator',task_id:'call1'},
            {type:'node_output',node_id:'tools:tool:calculator',run_id:'call1',status:'ok'},
            {type:'node_output',node_id:'tools',run_id:'dispatcher',status:'ok'}];
        const {nodeStatuses}=computeStatuses(events,new Set(['tools']));
        expect(nodeStatuses.get('tools:tool:calculator')).toBe('completed');
        expect(nodeStatuses.get('tools:tool:unused')).toBeUndefined();
    });
});
