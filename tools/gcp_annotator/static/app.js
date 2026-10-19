/* Buildless client; all editable point positions are canonical TEST pixels. */
'use strict';
const G=window.GCPGeometry, $=id=>document.getElementById(id);
const labels={not_started:'未开始',in_progress:'标注中',completed:'已完成',skipped:'跳过'};
const colors=['#ffcd65','#7fe3f2','#df9cff','#94e3a8','#ff9f9f','#9cbcff','#f4a4d1','#c6e680'];
let catalog,doc=null,meta=null,selected=null,history=new G.History(),dirty=false,change=0;
let saving=null,saveTimer=null,loading=false,space=false,recovery=null,messageTimer;
let mode='native';
const views={};
const pointColor=p=>colors[(Number(p.id.slice(4))-1)%colors.length];
const localKey=id=>'mhinet-ground-gcp:'+catalog.output+':'+id;
function tell(message,success=false){$('message').textContent=message;$('message').className=success?'success':'';$('message').hidden=false;clearTimeout(messageTimer);messageTimer=setTimeout(()=>$('message').hidden=true,9000);}
async function api(url,options={}){
  const r=await fetch(url,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
  if(!r.ok){let m;try{m=(await r.json()).detail;}catch{m=r.statusText;}throw new Error(typeof m==='string'?m:JSON.stringify(m));}
  return r.json();
}
function editable(){return doc&&!loading&&!recovery&&!$('reviewMode').checked;}
function stash(){try{localStorage.setItem(localKey(doc.pair_id),JSON.stringify({payload:doc,savedAt:Date.now()}));}catch{tell('浏览器本地备份不可用，请确认服务器保存成功');}}
function changed(){dirty=true;change++;stash();$('saveState').textContent='等待自动保存…';clearTimeout(saveTimer);saveTimer=setTimeout(()=>flush().catch(e=>tell(e.message)),500);}
function edit(fn,geometry=true){
  if(!doc||loading||recovery)return;
  history.push(doc);fn();
  if(geometry){doc.review={status:'unreviewed',reviewer:doc.review.reviewer,checked_ids:[]};if(doc.status==='completed'||doc.status==='not_started')doc.status='in_progress';}
  changed();render();
}
async function flush(){
  clearTimeout(saveTimer);
  if(saving)return saving;
  if(!doc||!dirty)return;
  saving=(async()=>{
    while(dirty){
      const version=change, payload=G.clone(doc), pair=doc.pair_id;
      $('saveState').textContent='正在保存…';
      try{
        const saved=await api('/api/pairs/'+encodeURIComponent(pair),{method:'PUT',body:JSON.stringify(payload)});
        doc.revision=saved.revision;doc.updated_at=saved.updated_at;doc.created_at=saved.created_at;
        if(version===change){doc=saved;dirty=false;try{localStorage.removeItem(localKey(pair));}catch{};}
        else stash();
        const row=catalog.pairs.find(p=>p.pair_id===pair);
        Object.assign(row,{status:saved.status,points:saved.points.length,review:saved.review.status});
        $('saveState').textContent='已保存 · '+new Date(saved.updated_at).toLocaleTimeString();
        renderList();
      }catch(e){$('saveState').textContent='保存失败 · 本地修改已保留';throw e;}
    }
  })();
  try{await saving;}finally{saving=null;}
}
function visiblePairs(){
  let rows=$('scope').value==='sample'?catalog.selection.pair_ids.map(id=>catalog.pairs.find(p=>p.pair_id===id)):catalog.pairs;
  const term=$('search').value.trim().toLowerCase(), filter=$('filter').value;
  return rows.filter(p=>p.pair_id.toLowerCase().includes(term)&&(filter==='all'||p.status===filter));
}
function renderList(){
  if(!catalog)return;
  const list=$('pairList');list.replaceChildren();
  for(const p of visiblePairs()){
    const button=document.createElement('button');button.className='pair'+(doc?.pair_id===p.pair_id?' active':'');
    button.textContent=p.pair_id;const small=document.createElement('small');small.textContent=labels[p.status]+' · '+p.points+' 组'+(p.review==='approved'?' · 已审核':'');button.append(small);
    button.onclick=()=>openPair(p.pair_id).catch(e=>tell(e.message));list.append(button);
  }
  const rows=$('scope').value==='sample'?catalog.pairs.filter(p=>p.selected):catalog.pairs;
  const count=s=>rows.filter(p=>p.status===s).length;
  $('progress').textContent=`${rows.length} 对 · seed ${catalog.selection.seed}\n完成 ${count('completed')} / 标注中 ${count('in_progress')} / 跳过 ${count('skipped')} / 未开始 ${count('not_started')}`;
  $('outputPath').textContent='服务器数据目录：'+catalog.output;
}
async function loadImages(){
  await Promise.all(Object.values(views).map(async v=>{
    const image=new Image();image.src=`/api/pairs/${encodeURIComponent(doc.pair_id)}/image/${v.side}/${mode}`;
    await image.decode();v.image=image;v.fit();
  }));
}
async function openPair(id){
  if(loading||recovery||doc?.pair_id===id)return;
  loading=true;document.body.classList.add('loading');
  try{
    await flush();
    const result=await api('/api/pairs/'+encodeURIComponent(id));
    doc=result.annotation;meta=result.metadata;selected=null;dirty=false;history=new G.History();
    $('pairTitle').textContent=id;
    await loadImages();
    try{
      const local=JSON.parse(localStorage.getItem(localKey(id))||'null');
      if(local){recovery=local;$('recovery').hidden=false;$('restoreLocal').disabled=local.payload.revision!==doc.revision;
        $('recoveryText').textContent=local.payload.revision!==doc.revision?'发现本地修改，但服务器版本已更新。请下载恢复 JSON 人工核对，再使用服务器版本。':'发现尚未同步的本地修改。';}
    }catch{tell('浏览器本地备份无法读取；服务器数据已正常载入');}
    $('saveState').textContent='已载入 · 版本 '+doc.revision;
    render();renderList();
  }finally{loading=false;document.body.classList.remove('loading');}
}
function toView(side,p){return mode==='test'?p:G.project(meta.images[side].test_to_native,p);}
function fromView(side,p){return G.snapBounds(mode==='test'?p:G.project(meta.images[side].native_to_test,p),meta.images[side].test_size);}
function legal(side,p){
  return G.inside(p,meta.images[side].test_size)&&G.inside(G.project(meta.images[side].test_to_native,p),meta.images[side].native_size);
}
class Viewer{
  constructor(side){
    this.side=side;this.canvas=$('canvas'+side);this.ctx=this.canvas.getContext('2d');this.v={x:0,y:0,scale:1};this.image=null;this.pointer=null;this.drag=null;
    new ResizeObserver(()=>{this.resize();}).observe(this.canvas);
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.pointer=this.screen(e);this.v=G.zoomAt(this.v,this.pointer,Math.exp(-e.deltaY*.0015));this.draw();},{passive:false});
    this.canvas.addEventListener('pointerdown',e=>this.down(e));this.canvas.addEventListener('pointermove',e=>this.move(e));this.canvas.addEventListener('pointerup',e=>this.up(e));
    this.canvas.addEventListener('pointercancel',()=>this.cancel());
    this.canvas.addEventListener('pointerleave',()=>{if(!this.drag){this.pointer=null;this.draw();}});
    this.canvas.addEventListener('contextmenu',e=>e.preventDefault());
  }
  size(){const r=this.canvas.getBoundingClientRect();return [r.width,r.height];}
  resize(){const [w,h]=this.size(),dpr=window.devicePixelRatio||1;this.canvas.width=Math.round(w*dpr);this.canvas.height=Math.round(h*dpr);this.draw();}
  fit(){if(!this.image)return;const [w,h]=this.size(),scale=Math.min((w-24)/this.image.width,(h-24)/this.image.height);this.v={scale,x:(w-this.image.width*scale)/2+.5*scale,y:(h-this.image.height*scale)/2+.5*scale};this.draw();}
  screen(e){const r=this.canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
  hit(s){if(!doc)return null;let best=null,dist=13;for(const p of doc.points){const q=G.imageToScreen(toView(this.side,p[this.side.toLowerCase()]),this.v),d=Math.hypot(q[0]-s[0],q[1]-s[1]);if(d<dist){best=p;dist=d;}}return best;}
  down(e){
    if(!doc||loading||recovery)return;
    this.canvas.focus();this.canvas.setPointerCapture(e.pointerId);
    const s=this.screen(e),hit=this.hit(s),pan=e.button!==0||space;
    if(hit&&!pan){selected=hit.id;render();}
    this.drag={start:s,last:s,view:{...this.v},hit:!pan&&hit?hit.id:null,pan, moved:false, before:G.clone(doc)};
  }
  move(e){
    const s=this.screen(e);this.pointer=s;
    if(this.drag){
      const d=this.drag,dx=s[0]-d.start[0],dy=s[1]-d.start[1];if(Math.hypot(dx,dy)>3)d.moved=true;
      if(d.moved){
        if(d.hit&&!d.pan&&editable()){
          const p=fromView(this.side,G.screenToImage(s,this.v));
          if(legal(this.side,p)){
            if(!d.pointEdited){history.push(d.before);d.pointEdited=true;}
            doc.points.find(x=>x.id===d.hit)[this.side.toLowerCase()]=p;
            doc.review={status:'unreviewed',reviewer:doc.review.reviewer,checked_ids:[]};doc.status='in_progress';
            changed();drawAll();
          }
        }else if(!d.hit||d.pan){this.v={...d.view,x:d.view.x+dx,y:d.view.y+dy};}
      }
      d.last=s;
    }
    const view=G.screenToImage(s,this.v),p=meta?fromView(this.side,view):view;
    $('coord'+this.side).textContent=`测试 ${p[0].toFixed(2)}, ${p[1].toFixed(2)}`+(mode==='native'?` · 原貌 ${view[0].toFixed(1)}, ${view[1].toFixed(1)}`:'');
    this.draw();
  }
  up(e){
    if(!this.drag)return;const d=this.drag;this.drag=null;
    if(d.pointEdited){
      render();
    }else if(!d.moved&&!d.hit&&!d.pan&&editable()){
      const p=fromView(this.side,G.screenToImage(this.screen(e),this.v));
      if(!legal(this.side,p)){tell('该位置不在此侧测试影像的有效范围内，请在黄色范围内选择');return;}
      if(this.side==='B'&&!doc.draft){tell('请先在 A 图点击一个地面特征点');return;}
      edit(()=>{G.addClick(doc,this.side,p);selected=doc.draft?null:doc.points.at(-1).id;});
    }
    this.canvas.releasePointerCapture(e.pointerId);
  }
  cancel(){if(this.drag?.pointEdited){const revision=doc.revision;doc=this.drag.before;doc.revision=revision;changed();}this.drag=null;render();}
  center(p){const q=toView(this.side,p),[w,h]=this.size();this.v.scale=Math.max(this.v.scale,mode==='native'?1.5:3);this.v.x=w/2-q[0]*this.v.scale;this.v.y=h/2-q[1]*this.v.scale;this.draw();}
  draw(){
    const c=this.ctx,[w,h]=this.size(),dpr=window.devicePixelRatio||1;c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
    if(!this.image||!doc||!meta)return;
    const v=this.v;c.save();c.translate(v.x,v.y);c.scale(v.scale,v.scale);c.imageSmoothingEnabled=true;c.drawImage(this.image,-.5,-.5);c.restore();
    if(mode==='native'&&$('footprint').checked){
      const size=meta.images[this.side].test_size;
      const corners=[[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]].map(p=>G.imageToScreen(toView(this.side,p),v));
      c.save();c.beginPath();c.rect(0,0,w,h);c.moveTo(...corners[0]);corners.slice(1).forEach(p=>c.lineTo(...p));c.closePath();c.fillStyle='#060b13a0';c.fill('evenodd');c.beginPath();c.moveTo(...corners[0]);corners.slice(1).forEach(p=>c.lineTo(...p));c.closePath();c.strokeStyle='#f6d37d';c.lineWidth=1.5;c.stroke();c.restore();
    }
    for(const p of doc.points)this.marker(p[this.side.toLowerCase()],p.id,pointColor(p),p.id===selected,p.needs_review);
    if(this.side==='A'&&doc.draft)this.marker(doc.draft.a,'待配对','#fff',true,true);
    if(this.pointer){
      const [x,y]=this.pointer;c.strokeStyle='#fff9';c.lineWidth=1;c.beginPath();c.moveTo(x-20,y);c.lineTo(x-6,y);c.moveTo(x+6,y);c.lineTo(x+20,y);c.moveTo(x,y-20);c.lineTo(x,y-6);c.moveTo(x,y+6);c.lineTo(x,y+20);c.stroke();
      if($('magnifier').checked)this.loupe(x,y);
    }
  }
  marker(p,id,color,active,review){
    const c=this.ctx,[x,y]=G.imageToScreen(toView(this.side,p),this.v);c.save();c.strokeStyle=color;c.fillStyle=color;c.lineWidth=active?2.5:1.5;c.setLineDash(review?[3,2]:[]);c.beginPath();c.arc(x,y,active?9:6,0,Math.PI*2);c.stroke();c.setLineDash([]);c.beginPath();c.moveTo(x-4,y);c.lineTo(x+4,y);c.moveTo(x,y-4);c.lineTo(x,y+4);c.stroke();c.font='bold 11px system-ui';const text=id.replace('GCP-',''),tw=c.measureText(text).width;c.fillStyle='#09131dde';c.fillRect(x+10,y-19,tw+8,17);c.fillStyle=color;c.fillText(text,x+14,y-6);c.restore();
  }
  loupe(x,y){
    const c=this.ctx,[w,h]=this.size(),r=62,cx=Math.max(r+4,Math.min(w-r-4,x+90)),cy=Math.max(r+4,Math.min(h-r-4,y-90)),q=G.screenToImage([x,y],this.v),zoom=Math.max(this.v.scale*4,4);
    c.save();c.beginPath();c.arc(cx,cy,r,0,Math.PI*2);c.clip();c.fillStyle='#0b131d';c.fillRect(cx-r,cy-r,r*2,r*2);c.imageSmoothingEnabled=false;c.drawImage(this.image,q[0]+.5-r/zoom,q[1]+.5-r/zoom,2*r/zoom,2*r/zoom,cx-r,cy-r,r*2,r*2);c.strokeStyle='#ffdb68';c.lineWidth=1;c.beginPath();c.moveTo(cx-10,cy);c.lineTo(cx+10,cy);c.moveTo(cx,cy-10);c.lineTo(cx,cy+10);c.stroke();c.restore();c.strokeStyle='#bfd6e3';c.beginPath();c.arc(cx,cy,r,0,Math.PI*2);c.stroke();
  }
}
function drawAll(){Object.values(views).forEach(v=>v.draw());}
function focusPoint(p){selected=p.id;Object.values(views).forEach(v=>v.center(p[v.side.toLowerCase()]));render();}
function renderTable(){
  const rows=$('pointRows');rows.replaceChildren();if(!doc)return;
  for(const p of doc.points){
    const tr=document.createElement('tr');if(p.id===selected)tr.className='selected';tr.dataset.id=p.id;
    const label=document.createElement('td');label.textContent=p.id;label.style.color=pointColor(p);label.style.cursor='pointer';label.onclick=()=>focusPoint(p);tr.append(label);
    for(const side of ['a','b'])for(let i=0;i<2;i++){
      const td=document.createElement('td'),input=document.createElement('input');input.type='number';input.step='any';input.value=p[side][i].toFixed(4);input.min=0;input.max=meta.images[side.toUpperCase()].test_size[i]-1;input.disabled=$('reviewMode').checked;
      input.setAttribute('aria-label',`${p.id} ${side.toUpperCase()} ${i?'Y':'X'}`);
      input.onchange=()=>{const current=doc.points.find(q=>q.id===p.id),value=input.valueAsNumber,pos=current[side].slice();pos[i]=value;if(!legal(side.toUpperCase(),pos)){tell('坐标超出有效范围');input.value=current[side][i].toFixed(4);return;}edit(()=>doc.points.find(q=>q.id===p.id)[side]=pos);};td.append(input);tr.append(td);
    }
    const reviewCell=document.createElement('td'),flag=document.createElement('input');flag.type='checkbox';flag.checked=p.needs_review;flag.disabled=$('reviewMode').checked;flag.setAttribute('aria-label',p.id+' 待复核');flag.onchange=()=>edit(()=>doc.points.find(q=>q.id===p.id).needs_review=flag.checked);reviewCell.append(flag);tr.append(reviewCell);
    const checkCell=document.createElement('td'),check=document.createElement('input');check.type='checkbox';check.checked=doc.review.checked_ids.includes(p.id);check.disabled=!$('reviewMode').checked;check.setAttribute('aria-label',p.id+' 审核已看');check.onchange=()=>edit(()=>{doc.review.checked_ids=check.checked?[...new Set([...doc.review.checked_ids,p.id])]:doc.review.checked_ids.filter(id=>id!==p.id);doc.review.status='in_review';},false);checkCell.append(check);tr.append(checkCell);
    const action=document.createElement('td'),del=document.createElement('button');del.textContent='删除';del.disabled=$('reviewMode').checked;del.onclick=()=>removePoint(p.id);action.append(del);tr.append(action);
    tr.addEventListener('click',e=>{if(e.target===tr||e.target.tagName==='TD')focusPoint(p);});rows.append(tr);
  }
}
function render(){
  if(!doc)return;
  $('guidance').textContent=$('reviewMode').checked?'审核模式：逐点点击编号检查 A / B 两侧，勾选“审核已看”，全部确认后通过审核。':doc.draft?'A 点已记录，请在 B 图独立点击对应地面位置。Esc 取消，重新点击 A 可更换待配对点。':'先点 A，再点 B。请选择道路交叉点、地面标线和铺装角点，避开楼顶、树冠和车辆。';
  $('pointCount').textContent=doc.points.length+' 组 / 建议 12–16 组'+(doc.draft?' · 1 个待配对草稿':'');
  const ranges=['a','b'].map(side=>{const ps=doc.points.map(p=>p[side]),size=meta.images[side.toUpperCase()].test_size;return ps.length?((Math.max(...ps.map(p=>p[0]))-Math.min(...ps.map(p=>p[0])))/(size[0]-1)*(Math.max(...ps.map(p=>p[1]))-Math.min(...ps.map(p=>p[1])))/(size[1]-1)*100).toFixed(0):0;});
  $('coverage').textContent=`点集包围框占图面积 A ${ranges[0]}% / B ${ranges[1]}%（仅供分散度参考）`;
  $('annotator').value=doc.annotator;$('status').value=doc.status;$('skipReason').value=doc.skip_reason;$('notes').value=doc.notes;$('reviewer').value=doc.review.reviewer;
  $('reviewState').textContent={unreviewed:'未审核',in_review:'审核中',approved:'审核通过'}[doc.review.status];
  $('undo').disabled=!history.undoStack.length||$('reviewMode').checked;$('redo').disabled=!history.redoStack.length||$('reviewMode').checked;
  ['status','annotator','notes','skipReason'].forEach(id=>$(id).disabled=$('reviewMode').checked);
  $('delete').disabled=!selected||$('reviewMode').checked;$('cancelDraft').disabled=!doc.draft||$('reviewMode').checked;$('approve').disabled=!$('reviewMode').checked;
  renderTable();drawAll();
}
function removePoint(id=selected){if(!editable()||!id)return;edit(()=>{doc.points=doc.points.filter(p=>p.id!==id);selected=null;});}
function undo(redo=false){if(!editable())return;const value=history[redo?'redo':'undo'](doc);if(!value)return;const revision=doc.revision;doc=value;doc.revision=revision;doc.review={status:'unreviewed',reviewer:doc.review.reviewer,checked_ids:[]};if(doc.status==='completed')doc.status='in_progress';selected=null;changed();render();}
async function navigate(delta){const rows=visiblePairs(),i=rows.findIndex(p=>p.pair_id===doc?.pair_id);const target=rows[i+delta];if(target)await openPair(target.pair_id);}
$('scope').onchange=renderList;$('search').oninput=renderList;$('filter').onchange=renderList;
$('prev').onclick=()=>navigate(-1).catch(e=>tell(e.message));$('next').onclick=()=>navigate(1).catch(e=>tell(e.message));
$('save').onclick=()=>flush().then(()=>tell('服务器保存成功',true)).catch(e=>tell(e.message));
$('fit').onclick=()=>Object.values(views).forEach(v=>v.fit());
$('magnifier').onchange=drawAll;$('footprint').onchange=drawAll;
$('mode').onchange=async()=>{if(loading)return;loading=true;document.body.classList.add('loading');const before=mode;try{mode=$('mode').value;await loadImages();}catch(e){mode=before;$('mode').value=before;await loadImages();tell(e.message);}finally{loading=false;document.body.classList.remove('loading');}};
$('undo').onclick=()=>undo();$('redo').onclick=()=>undo(true);$('delete').onclick=()=>removePoint();$('cancelDraft').onclick=()=>{if(editable()&&doc.draft)edit(()=>doc.draft=null);};
for(const [id,key] of [['annotator','annotator'],['skipReason','skip_reason'],['notes','notes']])$(id).onchange=()=>{if(editable())edit(()=>doc[key]=$(id).value,false);};
$('status').onchange=()=>{
  if(!editable())return;const status=$('status').value;
  if(status==='not_started'&&(doc.points.length||doc.draft)){tell('已有控制点或草稿，请保持“标注中”或完成标注');$('status').value=doc.status;return;}
  if(status==='completed'&&(!doc.points.length||doc.draft||doc.points.some(p=>p.needs_review))){tell('完成前请补全点对、取消草稿并处理待复核点');$('status').value=doc.status;return;}
  if(status==='skipped'&&!doc.skip_reason.trim()){tell('请先填写无法可靠标注的原因');$('status').value=doc.status;return;}
  edit(()=>{doc.status=status;doc.review.status='unreviewed';doc.review.checked_ids=[];},false);
};
$('reviewMode').onchange=()=>{document.body.classList.toggle('review',$('reviewMode').checked);render();};
$('reviewer').onchange=()=>edit(()=>{doc.review.reviewer=$('reviewer').value;doc.review.status='in_review';},false);
$('approve').onclick=()=>{
  if(!doc||!$('reviewMode').checked)return;
  if(doc.status!=='completed'||!doc.review.reviewer.trim()||doc.points.some(p=>!doc.review.checked_ids.includes(p.id))){tell('需要已完成的影像对、审核者，并逐点检查全部控制点');return;}
  edit(()=>doc.review.status='approved',false);flush().catch(e=>tell(e.message));
};
$('export').onclick=async()=>{try{await flush();const result=await api('/api/export',{method:'POST'});for(const kind of ['jsonl','csv']){const a=document.createElement('a');a.href=result[kind];a.download='ground_gcps.'+kind;a.click();}tell(`已导出 ${result.pairs} 对 / ${result.points} 个正式控制点（仅已完成影像对）`,true);}catch(e){tell(e.message);}};
$('restoreLocal').onclick=()=>{if(!recovery||recovery.payload.revision!==doc.revision)return;history.push(doc);doc=recovery.payload;recovery=null;$('recovery').hidden=true;changed();render();};
$('discardLocal').onclick=()=>{localStorage.removeItem(localKey(doc.pair_id));recovery=null;$('recovery').hidden=true;};
$('downloadLocal').onclick=()=>{if(!recovery)return;const url=URL.createObjectURL(new Blob([JSON.stringify(recovery.payload,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=doc.pair_id+'.local-recovery.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.addEventListener('beforeunload',e=>{if(dirty||saving){stash();e.preventDefault();e.returnValue='';}});
window.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();document.activeElement?.blur();flush().catch(x=>tell(x.message));return;}
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  const key=e.key.toLowerCase();if(e.code==='Space'){space=true;e.preventDefault();}
  if((e.ctrlKey||e.metaKey)&&key==='s'){e.preventDefault();flush().catch(x=>tell(x.message));}
  else if((e.ctrlKey||e.metaKey)&&key==='z'){e.preventDefault();undo(e.shiftKey);}
  else if((e.ctrlKey||e.metaKey)&&key==='y'){e.preventDefault();undo(true);}
  else if(key==='delete'||key==='backspace'){e.preventDefault();removePoint();}
  else if(key==='escape'){if(editable()&&doc.draft)edit(()=>doc.draft=null);}
  else if(key==='f')Object.values(views).forEach(v=>v.fit());
  else if(key==='['||key===']')navigate(key==='['?-1:1).catch(x=>tell(x.message));
});
window.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});window.addEventListener('blur',()=>space=false);
(async()=>{try{views.A=new Viewer('A');views.B=new Viewer('B');catalog=await api('/api/catalog');renderList();await openPair(catalog.selection.pair_ids[0]);}catch(e){tell('载入失败：'+e.message);$('saveState').textContent='载入失败';}})();
