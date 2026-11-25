/* Motion follows the source, gradient and adapter paths in the paper's Figure 3.
   The SVG below the overlay retains every original drawing and glyph position. */
(() => {
  'use strict';
  const svg=document.getElementById('method-svg');
  const panel=document.getElementById('overview');
  const play=document.getElementById('overview-play');
  const buttons=[...document.querySelectorAll('[data-step]')];
  const ns='http://www.w3.org/2000/svg';
  const details=[
    {title:'Aligned supervision, different visual evidence',description:'Real, 2D and eligible GS observations reuse the source action target. Their visual content changes while the associated state, action and instruction remain aligned.',equation:'Real · 2D · GS',note:'GS validity is computed from local coverage, visibility and alignment metadata.'},
    {title:'A stable reference update for the local policy',description:'All sources use the backbone’s native action objective. In the SmolVLA implementation, vision and connector LoRA parameters are trainable while the action expert stays frozen.',equation:'g<sub>base</sub> = g<sub>real</sub> + 0.75 g<sub>2D</sub>',note:'The source coefficients weight gradients; they are not fixed shares of the final update.'},
    {title:'A bounded correction from eligible GS views',description:'VistaMix restricts the GS gradient to perception parameters, removes strongly opposing components, applies validity weighting and caps the correction relative to the base gradient.',equation:'g<sub>VM</sub> = g<sub>base</sub> + E<sub>v</sub>(ĉ<sub>GS</sub>)',note:'The effective GS coefficient is 0.25 × mean validity × cap multiplier. Without an eligible correction, the base update is used.'},
    {title:'One adapter update per client',description:'Each client sends its LoRA update and sample count. The server applies sample-weighted FedAvg and returns the global adapter for the next round; trajectories and visual assets stay local.',equation:'Δ<sub>t</sub> = ∑<sub>i</sub> (n<sub>i</sub> / ∑<sub>j</sub> n<sub>j</sub>) Δ<sub>i,t</sub>',note:'Different local visual sources share the same adapter format and server aggregation rule.'}
  ];
  // Coordinates use the original 960 × 449.01237 viewBox, including its arrows.
  const stages=[
    {focus:[95,25,220,390],color:'#f1a9c1',colors:['#f1a9c1','#96bbff','#f6c574','#f1a9c1','#96bbff','#f6c574','#f6c574'],paths:[
      'M96.2 228.1C107 228.1 117.8 188.8 117.8 149.5S128.7 70.8 139.5 70.8',
      'M97.6 228.2C111.9 228.2 126.2 212.2 126.2 196.2S132.6 164.2 139 164.2',
      'M97.6 238.6C103.2 238.6 108.8 245.6 108.8 252.6S123.9 266.6 139 266.6',
      'M223 77.5H261V344','M223 164.2H246V344','M173.2 302V350','M207 381H238']},
    {focus:[275,24,322,201],color:'#96bbff',paths:['M291 345V90H310','M453 111H480','M453 139H480','M453 166H480','M467 220V230'],outline:[277,26,318,195]},
    {focus:[338,254,262,164],color:'#f6c574',colors:['#f6c574','#96bbff','#f1a9c1','#f6c574','#f6c574','#f6c574','#f6c574','#96bbff','#f1a9c1'],paths:['M385 252V263','M469 252V263','M553 252V263','M384 293V308','M384 347V360','M422 380H431','M506 380H517','M469 293V307','M553 347V360'],outline:[341,256,255,154]},
    {focus:[625,16,331,424],color:'#80d5cc',paths:['M595 380H629V86H666','M826 240H865V69H880','M865 240V170H880','M865 240V274H880','M865 274V389H880','M654 240H639V430H323V213'],outline:[654,20,189,417]}
  ];
  function el(name,attrs,parent){const e=document.createElementNS(ns,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
  const overlay=el('g',{'aria-hidden':'true','id':'method-motion'},svg);
  const focusDefs=el('defs',{},svg);
  const mask=el('mask',{id:'method-focus-mask',maskUnits:'userSpaceOnUse',x:0,y:0,width:960,height:449.01237},focusDefs);
  el('rect',{width:960,height:449.01237,fill:'#fff'},mask);
  const hole=el('rect',{rx:9,fill:'#000'},mask);
  const spotlight=el('g',{'aria-hidden':'true',id:'method-spotlight'},svg);svg.insertBefore(spotlight,overlay);
  const shade=el('rect',{width:960,height:449.01237,mask:'url(#method-focus-mask)',class:'spot-shade'},spotlight);
  const focusRing=el('rect',{rx:9,fill:'none',class:'focus-ring','stroke-width':2.5},spotlight);
  const tracks=[];
  stages.forEach((stage,i)=>{
    const group=el('g',{'class':'flow-stage','data-stage':i},overlay);
    if(stage.outline){const[x,y,width,height]=stage.outline;el('rect',{x,y,width,height,rx:6,class:'stage-outline',stroke:stage.color},group);}
    stage.paths.forEach((d,j)=>{
      const color=(stage.colors||[])[j]||stage.color;
      const path=el('path',{d,class:'flow-trail',stroke:color},group);
      const length=path.getTotalLength();
      const dot=el('circle',{r:3.3,fill:color,class:'flow-dot'},group);
      tracks.push({stage:i,path,dot,length,phase:j*.12});
    });
  });
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let running=!reduced.matches, pinned=false, visible=false, current=0, elapsed=0,last=0;
  const phone=matchMedia('(max-width:700px)');
  const frames=['95 22 220 395','275 22 324 205','338 254 261 164','647 16 205 258'];
  const layer=document.createElementNS(ns,'g');
  [...svg.children].filter(e=>!['defs','title','desc'].includes(e.tagName)).forEach(e=>layer.append(e));
  const clipDefs=el('defs',{},svg),clip=el('clipPath',{id:'method-stage-window',clipPathUnits:'userSpaceOnUse'},clipDefs);
  const viewport=el('rect',{},clip);layer.setAttribute('clip-path','url(#method-stage-window)');svg.append(layer);
  function frame(){const bounds=phone.matches&&!panel.classList.contains('expanded')?frames[current]:'0 0 960 449.01237';svg.setAttribute('viewBox',bounds);const[x,y,width,height]=bounds.split(' ').map(Number);for(const[k,v]of Object.entries({x,y,width,height}))viewport.setAttribute(k,v);}
  phone.addEventListener('change',frame);
  function select(index,manual=false){
    current=index;elapsed=0;
    if(manual){pinned=true;running=!reduced.matches;}
    const stage=stages[index],detail=details[index];
    panel.style.setProperty('--stage-color',stage.color);
    document.getElementById('method-detail').style.setProperty('--stage-color',stage.color);
    panel.dataset.stage=String(index);panel.dataset.pinned=String(pinned);
    buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    overlay.querySelectorAll('.flow-stage').forEach((g,i)=>g.classList.toggle('active',i===index));
    const[x,y,width,height]=stage.focus;
    for(const target of[hole,focusRing])for(const[k,v]of Object.entries({x,y,width,height}))target.setAttribute(k,v);
    focusRing.setAttribute('stroke',stage.color);shade.setAttribute('opacity',pinned?'.66':'.20');
    document.getElementById('step-title').textContent=detail.title;
    document.getElementById('overview-description').textContent=detail.description;
    document.getElementById('step-equation').innerHTML=detail.equation;
    document.getElementById('step-note').textContent=detail.note;
    frame();updateControl();
    if(manual&&!reduced.matches){
      focusRing.getAnimations().forEach(a=>a.cancel());
      focusRing.animate([{strokeWidth:5,opacity:.5},{strokeWidth:2.5,opacity:1}],{duration:500,easing:'ease-out'});
      const detailPanel=document.getElementById('method-detail');detailPanel.getAnimations().forEach(a=>a.cancel());
      detailPanel.animate([{opacity:.35,translate:'0 6px'},{opacity:1,translate:'0 0'}],{duration:300,easing:'ease-out'});
    }
  }
  function updateControl(){
    play.textContent=running?'Pause animation':'Play animation';play.setAttribute('aria-pressed',String(running));panel.dataset.playing=String(running);
    document.getElementById('overview-tour').setAttribute('aria-pressed',String(!pinned&&running));
  }
  function tick(time){const delta=last?Math.min(time-last,80):0;last=time;
    if(running&&visible&&!document.hidden){elapsed+=delta;if(!pinned&&elapsed>=6500)select((current+1)%4);}
    for(const t of tracks){if(t.stage!==current)continue;const phase=(elapsed/1800+t.phase)%1;const point=t.path.getPointAtLength(phase*t.length);t.dot.setAttribute('cx',point.x);t.dot.setAttribute('cy',point.y);t.path.setAttribute('stroke-dasharray',`8 ${Math.max(8,t.length/8)}`);t.path.setAttribute('stroke-dashoffset',-elapsed/45);}
    requestAnimationFrame(tick);
  }
  buttons.forEach((b,i)=>b.addEventListener('click',()=>select(i,true)));
  play.addEventListener('click',()=>{running=!running;updateControl();});
  document.getElementById('overview-tour').addEventListener('click',()=>{pinned=false;running=!reduced.matches;select(0);});
  reduced.addEventListener('change',()=>{if(reduced.matches){running=false;updateControl();}});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;},{threshold:.1}).observe(panel);
  document.querySelector('.overview-steps').addEventListener('keydown',event=>{
    const i=buttons.indexOf(document.activeElement);if(i<0)return;
    let n;if(event.key==='ArrowRight')n=(i+1)%4;else if(event.key==='ArrowLeft')n=(i+3)%4;else if(event.key==='Home')n=0;else if(event.key==='End')n=3;else return;
    event.preventDefault();buttons[n].focus();select(n,true);
  });
  const expand=document.getElementById('overview-expand'),close=document.getElementById('overview-close');
  function enlarged(on){panel.classList.toggle('expanded',on);document.body.classList.toggle('figure-open',on);close.hidden=!on;frame();expand.setAttribute('aria-expanded',String(on));if(on){close.focus();}else expand.focus();}
  expand.addEventListener('click',()=>enlarged(true));close.addEventListener('click',()=>enlarged(false));
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&panel.classList.contains('expanded'))enlarged(false);});
  select(0);requestAnimationFrame(tick);
})();
