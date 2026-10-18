import {expect,it} from 'vitest';
import type {Node,Edge} from '@xyflow/react';
import type {NodeData,EdgeData,ExecutionEvent} from '../../langgraphics-web/src/types';
import {toolFlow} from '../../langgraphics-web/src/toolFlow';
const node=(id:string,role:NodeData['role']='chain',parentId?:string):Node<NodeData>=>({id,parentId,position:{x:0,y:0},data:{label:id,role,status:'idle',handles:[],nodeType:id==='tools'?'subgraph':'node'}});
const nodes=[node('react','llm'),node('review','executor'),node('tools','executor'),node('tools:__start__','executor','tools'),node('tools:__end__','executor','tools'),node('tools:tool:calculator','tool','tools')];
const edges:Edge<EdgeData>[]=[{id:'in',source:'react',target:'tools'},{id:'out',source:'tools',target:'review'}];
const events:ExecutionEvent[]=[
 {type:'edge_active',source:'react',target:'tools',edge_id:'in',elapsed_ms:10,seq:1},
 {type:'edge_active',source:'tools:__start__',target:'tools:tool:calculator',edge_id:'call',kind:'tool_call',elapsed_ms:11,seq:2},
 {type:'edge_active',source:'tools:tool:calculator',target:'tools:__end__',edge_id:'result',kind:'observation',payload:'336',elapsed_ms:20,seq:3},
 {type:'edge_active',source:'tools',target:'review',edge_id:'out',elapsed_ms:21,seq:4},
];
it('shows abstraction -> named tool -> actual result recipient without ToolNode plumbing',()=>{
 const view=toolFlow(nodes,edges,events);
 expect(view.nodes.map(n=>n.id)).toEqual(['react','review','tools:tool:calculator']);
 expect(view.nodes[2].parentId).toBeUndefined();
 const flow=view.events.filter(e=>e.type==='edge_active');
 expect(flow.map(e=>[e.source,e.target,e.kind,e.elapsed_ms])).toEqual([
 ['react','tools:tool:calculator','tool_call',11],['tools:tool:calculator','review','observation',20]]);
 expect(flow[1].payload).toBe('336');
 expect(view.edges.find(e=>e.source==='tools:tool:calculator')?.data?.label).toBe('Tool output');
});
it('shows output immediately when produced, before the dispatcher continuation',()=>{
 const view=toolFlow(nodes,edges,events.slice(0,3));
 expect(view.events.filter(e=>e.type==='edge_active').length).toBe(2);
 expect(view.events.find(e=>e.type==='edge_active'&&e.kind==='observation')?.elapsed_ms).toBe(20);
});
it('preserves each output production timestamp and does not mutate input edges',()=>{
 const second={...events[2],source:'tools:tool:calculator',payload:'parallel',elapsed_ms:20.5} as ExecutionEvent;
 const view=toolFlow(nodes,edges,[...events.slice(0,3),second,events[3]]);
 expect(view.events.filter(e=>e.type==='edge_active'&&e.kind==='observation').map(e=>e.elapsed_ms)).toEqual([20,20.5]);
 expect(edges.every(e=>e.data===undefined)).toBe(true);
});

it('does not guess the recipient when routing is unresolved, but still records output production',()=>{
 const view=toolFlow(nodes,[...edges,{id:'other',source:'tools',target:'react'}],events.slice(0,3));
 const result=view.events.find(e=>e.type==='edge_active'&&e.kind==='observation');
 expect(result?.elapsed_ms).toBe(20);
 expect(result?.type==='edge_active'&&result.target).toBe('tools:tool:calculator');
});
