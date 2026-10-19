/* All screen coordinates are CSS pixels. Canvas DPR never enters point data.
 * Image draw edges are at -0.5, so integer image coordinates are pixel centers.
 */
(function (root) {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  function project(H, p) {
    const w = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-12) throw new Error('投影点无效');
    return [(H[0][0]*p[0]+H[0][1]*p[1]+H[0][2])/w, (H[1][0]*p[0]+H[1][1]*p[1]+H[1][2])/w];
  }
  function screenToImage(p, v) { return [(p[0]-v.x)/v.scale, (p[1]-v.y)/v.scale]; }
  function imageToScreen(p, v) { return [p[0]*v.scale+v.x, p[1]*v.scale+v.y]; }
  function zoomAt(v, screen, factor) {
    const p=screenToImage(screen,v), scale=Math.min(100,Math.max(.01,v.scale*factor));
    return {scale,x:screen[0]-p[0]*scale,y:screen[1]-p[1]*scale};
  }
  function inside(p,size) { return p.every((x,i)=>Number.isFinite(x)&&x>=-1e-7&&x<=size[i]-1+1e-7); }
  function snapBounds(p,size) { return p.map((x,i)=>x<0&&x>=-1e-7?0:x>size[i]-1&&x<=size[i]-1+1e-7?size[i]-1:x); }
  function nextId(points) { return 'GCP-'+String(1+Math.max(0,...points.map(p=>Number(p.id.slice(4))))).padStart(3,'0'); }
  function addClick(doc,side,p) {
    if(side==='A') {doc.draft={a:p.slice()};return;}
    if(!doc.draft) throw new Error('请先在 A 图点击一个地面特征点');
    doc.points.push({id:nextId(doc.points),a:doc.draft.a.slice(),b:p.slice(),type:'ground',needs_review:false,note:''});
    doc.draft=null;
  }
  function randomPairs(seed,mapper,existing=[],count=20) {
    // Seeded, spread-out VIEW positions: no image matching or A-to-B H input.
    let state=seed>>>0;
    const random=()=>{state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
    const distance=(p,q)=>(p[0]-q[0])**2+(p[1]-q[1])**2;
    const near=(p,q)=>distance(p.a,q.a)<4||distance(p.b,q.b)<4;
    const candidates=[];
    for(let i=0;i<12000&&candidates.length<600;i++){
      const uv=[.04+.92*random(),.04+.92*random()],pair=mapper(uv);
      if(pair&&!existing.some(p=>near(pair,p)))candidates.push({uv,pair});
    }
    const selected=[],scores=candidates.map(()=>Infinity);
    for(let n=0;n<count;n++){
      let best=-1;
      for(let i=0;i<candidates.length;i++){
        if(selected.length){const last=selected.at(-1);scores[i]=Math.min(scores[i],distance(candidates[i].uv,last.uv));if(near(candidates[i].pair,last.pair))scores[i]=-1;}
        if(scores[i]>=0&&(best<0||scores[i]>scores[best]))best=i;
      }
      if(best<0)throw new Error('当前可见有效重叠区域不足，请恢复完整视图或缩小后再随机撒点');
      selected.push(candidates[best]);scores[best]=-1;
    }
    return selected.map(p=>p.pair);
  }
  class History {
    constructor(){this.undoStack=[];this.redoStack=[];}
    push(doc){this.undoStack.push(clone(doc));if(this.undoStack.length>200)this.undoStack.shift();this.redoStack=[];}
    undo(doc){if(!this.undoStack.length)return null;this.redoStack.push(clone(doc));return this.undoStack.pop();}
    redo(doc){if(!this.redoStack.length)return null;this.undoStack.push(clone(doc));return this.redoStack.pop();}
  }
  root.GCPGeometry={clone,project,screenToImage,imageToScreen,zoomAt,inside,snapBounds,nextId,addClick,randomPairs,History};
  if(typeof module!=='undefined') module.exports=root.GCPGeometry;
})(typeof globalThis!=='undefined'?globalThis:this);
