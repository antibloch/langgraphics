import {describe, expect, it} from 'vitest';
import {handoffsBetween, replayDuration, outputGlowsAt, OUTPUT_GLOW_MS} from '../../langgraphics-web/src/trace';
import {PlaybackClock} from '../../langgraphics-web/src/playback';
import {computeStatuses} from '../../langgraphics-web/src/hooks/useGraphState';
import type {ExecutionEvent} from '../../langgraphics-web/src/types';

const events: ExecutionEvent[] = [
    {type:'run_start',run_id:'actual',trace_source:'langgraph_scheduler_writes',elapsed_ms:0,seq:1},
    {type:'edge_active',source:'__start__',target:'agent',edge_id:'a',elapsed_ms:10,seq:2},
    {type:'node_start',node:'agent',task_id:'r1',elapsed_ms:11,seq:3},
    {type:'node_output',node_id:'agent',run_id:'r1',status:'ok',elapsed_ms:3000,seq:4},
    {type:'edge_active',source:'agent',target:'tool',edge_id:'b',elapsed_ms:3001,seq:5},
    {type:'node_start',node:'tool',task_id:'t1',elapsed_ms:3002,seq:6},
    {type:'node_output',node_id:'tool',run_id:'t1',status:'ok',elapsed_ms:3010,seq:7},
    {type:'run_end',run_id:'actual',elapsed_ms:8000,seq:8},
];
const visibleAt = (time: number) => events.filter(e => (e.elapsed_ms ?? 0) <= time);

describe('actual execution timing', () => {
    it('renders handoffs only in the frame crossing the actual timestamp, without a travel tail', () => {
        expect(handoffsBetween(events,0,9)).toEqual([]);
        expect(handoffsBetween(events,9,10).map(e=>e.seq)).toEqual([2]);
        expect(handoffsBetween(events,10,11)).toEqual([]);
        expect(handoffsBetween(events,10,100)).toEqual([]);
        expect(handoffsBetween(events,3000,3002).map(e=>e.seq)).toEqual([5]);
        expect(replayDuration(events)).toBe(8000);
    });
    it('preserves parallel handoffs and sub-frame ordering without slowing them down', () => {
        const parallel: ExecutionEvent[] = [events[1],{...events[1],seq:9}, {...events[1],seq:10,elapsed_ms:10.1}];
        expect(handoffsBetween(parallel,9,16).map(e=>e.seq)).toEqual([2,9,10]);
    });
    it('runs at exactly one recorded millisecond per elapsed millisecond, including sparse live waits', () => {
        const clock=new PlaybackClock();
        clock.tick(100,events.slice(0,3));
        expect(clock.tick(2100,events.slice(0,3))).toBe(2000);
        expect(clock.tick(3100,events)).toBe(3000);
        expect(clock.tick(8100,events)).toBe(8000);
        expect(clock.playing).toBe(false);
        expect(clock.tick(10000,events)).toBe(8000);
    });
    it('excludes all paused wall time and resumes at 1× without skipping buffered events', () => {
        const clock=new PlaybackClock(); clock.tick(100,events);
        clock.toggle(1100,events); expect(clock.position).toBe(1000);
        clock.tick(20000,events); expect(clock.position).toBe(1000);
        clock.toggle(30000,events);
        expect(clock.tick(31000,events)).toBe(2000);
        expect(clock.tick(32000,events)).toBe(3000);
        expect(computeStatuses(visibleAt(clock.position)).nodeStatuses.get('tool')).toBeUndefined();
        clock.tick(32002,events);
        expect(computeStatuses(visibleAt(clock.position)).nodeStatuses.get('tool')).toBe('active');
        clock.tick(32010,events);
        expect(computeStatuses(visibleAt(clock.position)).nodeStatuses.get('tool')).toBe('completed');
    });
    it('does not accumulate time before the first runtime event and restarts completed playback at the prompt', () => {
        const clock=new PlaybackClock();clock.tick(1,[]);clock.tick(10000,[]);
        expect(clock.tick(20000,events)).toBe(0);
        clock.tick(28000,events);clock.toggle(30000,events);
        expect(clock.position).toBe(0);expect(clock.playing).toBe(true);
        expect(clock.tick(30010,events)).toBe(10);
    });
});

it('makes actual outputs visible with a fading afterglow without changing execution timing',()=>{
 const outputs:ExecutionEvent[]=[{type:'edge_active',kind:'observation',source:'tool',target:'agent',edge_id:'output',elapsed_ms:100}];
 expect(outputGlowsAt(outputs,99).size).toBe(0);
 expect(outputGlowsAt(outputs,100).get('tool')).toBe(1);
 expect(outputGlowsAt(outputs,600).get('tool')).toBe(.5);
 expect(outputGlowsAt(outputs,100+OUTPUT_GLOW_MS).size).toBe(0);
 expect(replayDuration(outputs)).toBe(100);
 expect(outputGlowsAt([{...outputs[0],kind:'tool_call'} as ExecutionEvent],100).size).toBe(0);
});
