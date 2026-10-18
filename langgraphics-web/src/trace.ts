import type {ExecutionEvent, NodeRole} from './types';

export const roleLabels: Record<NodeRole, string> = {
    input: 'PROMPT', output: 'ANSWER', llm: 'LLM', planner: 'PLANNER', executor: 'EXECUTOR',
    tool: 'TOOL', memory: 'MEMORY', router: 'ROUTER', chain: 'WORKFLOW',
};

export function eventTime(event: ExecutionEvent): number {return event.elapsed_ms ?? 0;}

// A handoff is a timestamped event, not a measured journey along a 3D curve.
// Show only events crossed this rendered frame; never invent travel duration.
export function handoffsBetween(events: ExecutionEvent[], previousMs: number, clockMs: number) {
    if (clockMs <= previousMs) return [];
    return events.filter(event => event.type === 'edge_active' && eventTime(event) > previousMs && eventTime(event) <= clockMs);
}

export function replayDuration(events: ExecutionEvent[]): number {
    return events.reduce((end, event) => Math.max(end, eventTime(event)), 0);
}

// A perceptual afterglow, not transit time: starts exactly at real output production.
export const OUTPUT_GLOW_MS = 1000;
export function outputGlowsAt(events: ExecutionEvent[], clockMs: number): Map<string, number> {
    const glows = new Map<string, number>();
    for (const event of events) {
        if (event.type !== 'edge_active' || event.kind !== 'observation') continue;
        const age = clockMs - eventTime(event);
        if (age < 0 || age >= OUTPUT_GLOW_MS) continue;
        glows.set(event.source, Math.max(glows.get(event.source) ?? 0, 1 - age / OUTPUT_GLOW_MS));
    }
    return glows;
}
