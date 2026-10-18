import {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import type {ColorMode} from '@xyflow/react';
import {useWebSocket} from './hooks/useWebSocket';
import {useGraphState} from './hooks/useGraphState';
import {GraphCanvas} from './components/GraphCanvas';
import {eventTime} from './trace';
import {PlaybackClock} from './playback';
import type {RankDir} from './layout';
import './index.css';

const params = new URLSearchParams(window.location.search);
const theme: ColorMode = params.get('theme') === 'light' ? 'light' : 'dark';
const direction: RankDir = params.get('direction') === 'LR' ? 'LR' : 'TB';
const wsURL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:${params.get('ws_port') ?? '8765'}`;

function Index() {
    const {topology, events} = useWebSocket(wsURL);
    const [clockMs, setClockMs] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const clock = useRef(new PlaybackClock());
    const recorded = useRef(events);
    recorded.current = events;
    const runId = events.find(event => event.type === 'run_start')?.run_id;
    useEffect(() => {
        clock.current = new PlaybackClock(); setClockMs(0); setIsPlaying(true);
    }, [runId]);
    useEffect(() => {
        let frame: number;
        const animate = (now: number) => {
            setClockMs(clock.current.tick(now, recorded.current));
            setIsPlaying(clock.current.playing);
            frame = requestAnimationFrame(animate);
        };
        frame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(frame);
    }, []);
    const displayEvents = events.filter(event => eventTime(event) <= clockMs);
    const {nodes, edges} = useGraphState(topology, displayEvents, direction);
    const togglePlayback = () => {
        clock.current.toggle(performance.now(), events);
        setClockMs(clock.current.position); setIsPlaying(clock.current.playing);
    };
    return <GraphCanvas nodes={nodes} edges={edges} events={displayEvents}
        onTogglePlayback={togglePlayback} initialColorMode={theme}
        isPlaying={isPlaying} clockMs={clockMs} initialRankDir={direction}/>;
}

createRoot(document.getElementById('root')!).render(<Index/>);
