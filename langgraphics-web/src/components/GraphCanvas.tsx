import {useEffect, useRef, useState} from 'react';
import type {ColorMode, Edge, Node} from '@xyflow/react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {CSS2DObject, CSS2DRenderer} from 'three/addons/renderers/CSS2DRenderer.js';
import {positions3D, pathNodes} from '../graph3d';
import {toolFlow} from '../toolFlow';
import type {EdgeData, ExecutionEvent, NodeData} from '../types';
import {roleLabels, handoffsBetween, outputGlowsAt} from '../trace';
import type {RankDir} from '../layout';

interface Props {
    nodes: Node<NodeData>[];
    edges: Edge<EdgeData>[];
    events: ExecutionEvent[];
    onTogglePlayback: () => void;
    isPlaying: boolean;
    clockMs: number;
    initialRankDir?: RankDir;
    initialColorMode?: ColorMode;
}
const colors = {idle: '#64748b', active: '#34d399', completed: '#60a5fa', traversed: '#60a5fa', error: '#fb7185'};

export function GraphCanvas({nodes, edges, events, onTogglePlayback, isPlaying, clockMs,
    initialColorMode = 'dark', initialRankDir = 'TB'}: Props) {
    const flow = toolFlow(nodes, edges, events);
    nodes = flow.nodes; edges = flow.edges; events = flow.events;
    const mount = useRef<HTMLDivElement>(null);
    const update = useRef<() => void>(() => {});
    const latest = useRef({nodes, edges, events, direction: initialRankDir});
    latest.current = {nodes, edges, events, direction: initialRankDir};
    const [failure, setFailure] = useState('');
    const animation = useRef({events, clockMs});
    animation.current = {events, clockMs};

    useEffect(() => {
        const host = mount.current!;
        let renderer: THREE.WebGLRenderer;
        try {renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});}
        catch {setFailure('3D needs WebGL. Enable hardware acceleration in your browser and reload.'); return;}
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.domElement.setAttribute('aria-label', 'Interactive 3D execution graph');
        renderer.domElement.setAttribute('role', 'img');
        host.appendChild(renderer.domElement);
        const labels = new CSS2DRenderer();
        labels.domElement.className = 'graph-label-layer';
        host.appendChild(labels.domElement);
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, 1, 1, 50000);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.minDistance = 50;
        controls.maxDistance = 30000;
        scene.add(new THREE.AmbientLight(0xffffff, 2));
        const light = new THREE.DirectionalLight(0xffffff, 3);
        light.position.set(100, 200, 500); scene.add(light);
        let group = new THREE.Group(); scene.add(group);
        let edgeMeshes = new Map<string, THREE.Mesh>();
        let previousClock = -1;
        let highlighted = new Set<string>();
        let outputDots = new Map<string, THREE.Mesh>();
        const glowCanvas = document.createElement('canvas'); glowCanvas.width = glowCanvas.height = 128;
        const glowContext = glowCanvas.getContext('2d')!;
        const gradient = glowContext.createRadialGradient(64,64,0,64,64,64);
        gradient.addColorStop(0,'rgba(255,190,190,1)');
        gradient.addColorStop(.15,'rgba(255,45,65,.95)');
        gradient.addColorStop(.45,'rgba(255,20,45,.4)');
        gradient.addColorStop(1,'rgba(255,0,30,0)');
        glowContext.fillStyle = gradient; glowContext.fillRect(0,0,128,128);
        const glowTexture = new THREE.CanvasTexture(glowCanvas);
        const makeOutputDot = () => {
            const dot = new THREE.Mesh(new THREE.SphereGeometry(8,16,12),
                new THREE.MeshBasicMaterial({color:'#ff304f',transparent:true,depthTest:false,depthWrite:false}));
            dot.renderOrder = 20; dot.visible = false;
            const halo = new THREE.Sprite(new THREE.SpriteMaterial({map:glowTexture,
                color:'#ffffff',transparent:true,depthTest:false,depthWrite:false,blending:THREE.AdditiveBlending}));
            halo.scale.set(100,100,1); halo.renderOrder = 19; dot.add(halo);
            return dot;
        };
        let active: THREE.Mesh[] = [];
        let topologyKey = '';
        const disposeGroup = () => {
            group.traverse(object => {
                if (object instanceof THREE.Mesh) {
                    object.geometry.dispose();
                    const materials = Array.isArray(object.material) ? object.material : [object.material];
                    materials.forEach(material => material.dispose());
                }
                if (object instanceof THREE.Sprite) object.material.dispose();
                if (object instanceof CSS2DObject) object.element.remove();
            });
            scene.remove(group);
        };
        const fit = () => {
            const box = new THREE.Box3().setFromObject(group);
            if (box.isEmpty()) return;
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3()).length();
            const distance = Math.max(260, size / (2 * Math.tan(Math.PI / 8)) / Math.min(camera.aspect, 1) * 1.0);
            controls.target.copy(center);
            camera.position.copy(center).add(new THREE.Vector3(.45, .2, 1).normalize().multiplyScalar(distance));
            controls.update();
        };
        update.current = () => {
            const {nodes, edges, direction} = latest.current;
            const positions = positions3D(nodes, edges, direction);
            const visited = pathNodes(edges);
            disposeGroup(); group = new THREE.Group(); scene.add(group);
            edgeMeshes = new Map(); outputDots = new Map(); active = [];
            for (const node of nodes) {
                const point = positions.get(node.id)!;
                const status = (node.className ?? (visited.has(node.id) ? 'completed' : node.data.status) ?? 'idle') as keyof typeof colors;
                const color = colors[status] ?? colors.idle;
                const radius = node.data.nodeType === 'subgraph' ? 28 : node.parentId ? (node.data.nodeType === 'start' || node.data.nodeType === 'end' ? 9 : 22) : 25;
                const role = node.data.role ?? 'chain';
                const geometry = role === 'tool' ? new THREE.BoxGeometry(radius * 1.6, radius * 1.6, radius * 1.6)
                    : role === 'llm' ? new THREE.OctahedronGeometry(radius * 1.3)
                    : role === 'planner' ? new THREE.TetrahedronGeometry(radius * 1.4)
                    : role === 'executor' ? new THREE.CylinderGeometry(radius, radius, radius * 1.7, 6)
                    : role === 'memory' ? new THREE.CylinderGeometry(radius, radius, radius * 1.7, 24)
                    : new THREE.SphereGeometry(radius, 24, 16);
                const sphere = new THREE.Mesh(geometry,
                    new THREE.MeshStandardMaterial({color, emissive: color, emissiveIntensity: status === 'active' ? .7 : .15,
                        metalness: .35, roughness: .3}));
                sphere.position.set(point.x, point.y, point.z); group.add(sphere);
                if (status === 'active') active.push(sphere);
                const label = document.createElement('div');
                label.className = `graph-node-label ${status} ${node.parentId ? 'nested-node' : ''} ${node.parentId && (node.data.nodeType === 'start' || node.data.nodeType === 'end') ? 'scope-port' : ''}`;
                const badge = document.createElement('strong'); badge.className = `role-badge role-${role}`;
                badge.textContent = role === 'tool' ? 'TOOL CALL' : roleLabels[role]; label.appendChild(badge);
                const title = document.createElement('span'); title.textContent = node.data.label || node.id; label.appendChild(title);
                const tag = new CSS2DObject(label); tag.center.set(.5, 0); tag.position.set(0, -45, 0); sphere.add(tag);
            }
            for (const edge of edges) {
                const from = positions.get(edge.source), to = positions.get(edge.target);
                if (!from || !to) continue;
                const a = new THREE.Vector3(from.x, from.y, from.z);
                const b = new THREE.Vector3(to.x, to.y, to.z);
                const middle = a.clone().lerp(b, .5);
                const reverse = edges.some(other => other.source === edge.target && other.target === edge.source);
                middle.z += reverse ? (edge.source < edge.target ? 90 : -90) : 30;
                if (edge.source === edge.target) {middle.x += 100; middle.z += 100; b.y += 3;}
                const curve = new THREE.QuadraticBezierCurve3(a, middle, b);
                const status = edge.data?.status ?? 'idle';
                const color = colors[status];
                const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, status === 'idle' ? .7 : 1.6, 6, false),
                    new THREE.MeshBasicMaterial({color, transparent: true, opacity: status === 'idle' ? .3 : .9}));
                tube.userData.baseColor = color; tube.userData.baseOpacity = status === 'idle' ? .3 : .9;
                group.add(tube); edgeMeshes.set(edge.id, tube);
                const arrow = new THREE.Mesh(new THREE.ConeGeometry(4, 11, 10), new THREE.MeshBasicMaterial({color}));
                arrow.position.copy(curve.getPoint(.82));
                arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(.82).normalize()); group.add(arrow);
                if (edge.data?.label === 'Tool output' && !outputDots.has(edge.source)) {
                    const dot = makeOutputDot();
                    dot.position.copy(curve.getPoint(.12)); group.add(dot); outputDots.set(edge.source,dot);
                }
            }
            for (const node of nodes.filter(node => node.data.role === 'tool')) {
                if (outputDots.has(node.id)) continue;
                const point = positions.get(node.id)!;
                const dot = makeOutputDot();
                dot.position.set(point.x + 38,point.y + 15,point.z);
                group.add(dot); outputDots.set(node.id,dot);
            }
            const key = direction + nodes.map(node => `${node.id}:${node.parentId ?? ''}`).join('|') + edges.map(edge => edge.id).join('|');
            if (key !== topologyKey) {topologyKey = key; fit();}
        };
        const resize = new ResizeObserver(() => {
            const {width, height} = host.getBoundingClientRect();
            camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix();
            renderer.setSize(width, height); labels.setSize(width, height);
        });
        resize.observe(host);
        camera.aspect = host.clientWidth / Math.max(host.clientHeight, 1);
        camera.updateProjectionMatrix();
        update.current();
        let frame = 0;
        const animate = () => {
            frame = requestAnimationFrame(animate);
            controls.autoRotate = false;
            controls.update();
            active.forEach(mesh => mesh.scale.setScalar(1 + Math.sin(animation.current.clockMs / 220) * .12));
            const state = animation.current;
            if (state.clockMs !== previousClock) {
                const handoffs = handoffsBetween(state.events, previousClock, state.clockMs);
                highlighted = new Set(handoffs.flatMap(event => event.type === 'edge_active' ? [event.edge_id] : []));
                previousClock = state.clockMs;
            }
            for (const [id, mesh] of edgeMeshes) {
                const material = mesh.material as THREE.MeshBasicMaterial;
                material.color.set(mesh.userData.baseColor); material.opacity = mesh.userData.baseOpacity;
                if (highlighted.has(id)) {material.color.set('#e7fff6'); material.opacity = 1;}
            }
            const glows = outputGlowsAt(state.events,state.clockMs);
            for (const [node, dot] of outputDots) {
                const strength = glows.get(node) ?? 0;
                dot.visible = strength > 0;
                (dot.material as THREE.MeshBasicMaterial).opacity = Math.sqrt(strength);
                (dot.children[0] as THREE.Sprite).material.opacity = Math.sqrt(strength);
            }
            renderer.render(scene, camera); labels.render(scene, camera);
        };
        frame = requestAnimationFrame(animate);
        return () => {
            cancelAnimationFrame(frame); resize.disconnect(); controls.dispose(); disposeGroup(); glowTexture.dispose(); renderer.dispose();
            renderer.domElement.remove(); labels.domElement.remove(); update.current = () => {};
        };
    }, []);
    useEffect(() => {update.current();}, [nodes, edges, events, initialRankDir]);
    const error = events.some(event => event.type === 'error');
    return <main className={`graph-app ${initialColorMode === 'light' ? 'light' : 'dark'}`}>
        <div ref={mount} className="graph-viewport" />
        <header className="graph-toolbar">
            <div className="graph-brand">LANGGRAPHICS <span>3D</span></div>
            <button className="playback-button" disabled={!events.length} onClick={onTogglePlayback}
                aria-label={isPlaying ? 'Pause' : 'Resume'}>
                <span aria-hidden="true">{isPlaying ? 'Ⅱ' : '▶'}</span>{isPlaying ? 'Pause' : 'Resume'}
            </button>
        </header>
        {(failure || error) && <div className="graph-status" role="status">{failure || 'Run failed'}</div>}
    </main>;
}
