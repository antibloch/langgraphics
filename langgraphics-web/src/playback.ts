import {replayDuration} from './trace';
import type {ExecutionEvent} from './types';

/** A monotonic 1× display clock. Pause never alters or delays agent execution. */
export class PlaybackClock {
    position = 0;
    playing = true;
    private previous: number | null = null;

    tick(now: number, events: ExecutionEvent[]): number {
        if (!events.length) {this.previous = null; return this.position;}
        const delta = this.previous === null ? 0 : Math.max(0, now - this.previous);
        this.previous = now;
        if (this.playing) this.position += delta;
        // Sparse live events do not stop the clock during real model/tool work.
        if (events.some(event => event.type === 'run_end' || event.type === 'error')) {
            const end = replayDuration(events);
            this.position = Math.min(this.position, end);
            if (this.position >= end) this.playing = false;
        }
        return this.position;
    }

    toggle(now: number, events: ExecutionEvent[]) {
        this.tick(now, events);
        if (!this.playing && events.some(event => event.type === 'run_end' || event.type === 'error') && this.position >= replayDuration(events)) {
            this.position = 0;
        }
        this.playing = !this.playing;
        this.previous = now;
    }
}
