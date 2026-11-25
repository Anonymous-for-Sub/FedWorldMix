/* Motion follows the source, gradient and adapter paths in the paper's Figure 3.
   The SVG below the overlay retains every original drawing and glyph position. */
(() => {
  'use strict';
  const svg=document.getElementById('method-svg');
  const panel=document.getElementById('overview');
  const play=document.getElementById('overview-play');
  const buttons=[...document.querySelectorAll('[data-step]')];
  const ns='http://www.w3.org/2000/svg';
  const descriptions=[
    'Real, 2D and eligible GS observations follow the three source paths into the shared local VLA.',
    'Vision and connector LoRA receive local updates; the action expert retains its frozen parameters.',
    'VistaMix restricts the GS gradient to perception parameters, projects conflicts, and bounds its contribution to the Real-plus-2D base.',
    'The client uploads its adapter update and sample count. FedAvg returns the global adapter for the next round.'
  ];
  // Coordinates use the original 960 × 449.01237 viewBox, including its arrows.
  const stages=[
    {color:'#f1a9c1',colors:['#f1a9c1','#96bbff','#f6c574','#f1a9c1','#96bbff','#f6c574','#f6c574'],paths:[
      'M96.2 228.1C107 228.1 117.8 188.8 117.8 149.5S128.7 70.8 139.5 70.8',
      'M97.6 228.2C111.9 228.2 126.2 212.2 126.2 196.2S132.6 164.2 139 164.2',
      'M97.6 238.6C103.2 238.6 108.8 245.6 108.8 252.6S123.9 266.6 139 266.6',
      'M223 77.5H261V344','M223 164.2H246V344','M173.2 302V350','M207 381H238']},
    {color:'#96bbff',paths:['M291 345V90H310','M453 111H480','M453 139H480','M453 166H480','M467 220V230'],outline:[277,26,318,195]},
    {color:'#f6c574',colors:['#f6c574','#96bbff','#f1a9c1','#f6c574','#f6c574','#f6c574','#f6c574','#96bbff','#f1a9c1'],paths:['M385 252V263','M469 252V263','M553 252V263','M384 293V308','M384 347V360','M422 380H431','M506 380H517','M469 293V307','M553 347V360'],outline:[341,256,255,154]},
    {color:'#80d5cc',paths:['M595 380H629V86H666','M826 240H865V69H880','M865 240V170H880','M865 240V274H880','M865 274V389H880','M654 240H639V430H323V213'],outline:[654,20,189,417]}
  ];
  function el(name,attrs,parent){const e=document.createElementNS(ns,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
  const overlay=el('g',{'aria-hidden':'true','id':'method-motion'},svg);
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
  let running=!reduced.matches, visible=false, current=0, elapsed=0,last=0;
  function select(index,manual=false){current=index;elapsed=0;if(manual)running=false;
    buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    overlay.querySelectorAll('.flow-stage').forEach((g,i)=>g.classList.toggle('active',i===index));
    document.getElementById('overview-description').textContent=descriptions[index];updateControl();}
  function updateControl(){play.textContent=running?'Pause animation':'Play animation';play.setAttribute('aria-pressed',String(running));panel.dataset.playing=String(running);}
  function tick(time){const delta=last?Math.min(time-last,80):0;last=time;
    if(running&&visible&&!document.hidden){elapsed+=delta;if(elapsed>=5000)select((current+1)%4);}
    for(const t of tracks){if(t.stage!==current)continue;const phase=(elapsed/1800+t.phase)%1;const point=t.path.getPointAtLength(phase*t.length);t.dot.setAttribute('cx',point.x);t.dot.setAttribute('cy',point.y);t.path.setAttribute('stroke-dasharray',`8 ${Math.max(8,t.length/8)}`);t.path.setAttribute('stroke-dashoffset',-elapsed/45);}
    requestAnimationFrame(tick);
  }
  buttons.forEach((b,i)=>b.addEventListener('click',()=>select(i,true)));
  play.addEventListener('click',()=>{running=!running;updateControl();});
  reduced.addEventListener('change',()=>{if(reduced.matches){running=false;updateControl();}});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;},{threshold:.1}).observe(panel);
  document.querySelector('.overview-steps').addEventListener('keydown',event=>{
    const i=buttons.indexOf(document.activeElement);if(i<0)return;
    let n;if(event.key==='ArrowRight')n=(i+1)%4;else if(event.key==='ArrowLeft')n=(i+3)%4;else if(event.key==='Home')n=0;else if(event.key==='End')n=3;else return;
    event.preventDefault();buttons[n].focus();select(n,true);
  });
  const expand=document.getElementById('overview-expand'),close=document.getElementById('overview-close');
  function enlarged(on){panel.classList.toggle('expanded',on);document.body.classList.toggle('figure-open',on);close.hidden=!on;if(on){close.focus();}else expand.focus();}
  expand.addEventListener('click',()=>enlarged(true));close.addEventListener('click',()=>enlarged(false));
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&panel.classList.contains('expanded'))enlarged(false);});
  select(0);requestAnimationFrame(tick);
})();
