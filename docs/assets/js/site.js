/* Theme, mobile navigation and paired views from the paper. */
(() => {
  'use strict';
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  function setTheme(theme){
    document.body.dataset.theme=theme;
    document.documentElement.style.colorScheme=theme;
    document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.themeChoice===theme)));
    try{localStorage.setItem('fedvista-theme',theme);}catch{}
  }
  let saved;try{saved=localStorage.getItem('fedvista-theme');}catch{}
  setTheme(saved==='light'?'light':'dark');
  document.querySelectorAll('[data-theme-choice]').forEach(b=>b.addEventListener('click',()=>setTheme(b.dataset.themeChoice)));
  const menu=document.getElementById('nav-toggle'),links=document.getElementById('nav-links');
  function closeMenu(){links.classList.remove('open');menu.setAttribute('aria-expanded','false');}
  menu.addEventListener('click',()=>{const on=links.classList.toggle('open');menu.setAttribute('aria-expanded',String(on));});
  links.querySelectorAll('a').forEach(a=>a.addEventListener('click',closeMenu));
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMenu();});
  function viewportClip(svg,id){
    const ns='http://www.w3.org/2000/svg';
    const group=document.createElementNS(ns,'g');
    [...svg.children].filter(e=>!['defs','title','desc'].includes(e.tagName)).forEach(e=>group.append(e));
    const defs=document.createElementNS(ns,'defs'),clip=document.createElementNS(ns,'clipPath'),rect=document.createElementNS(ns,'rect');
    clip.id=id;clip.setAttribute('clipPathUnits','userSpaceOnUse');clip.append(rect);defs.append(clip);svg.append(defs);group.setAttribute('clip-path',`url(#${id})`);svg.append(group);
    return (x,y,w,h)=>{for(const[k,v]of Object.entries({x,y,width:w,height:h}))rect.setAttribute(k,v);};
  }
  const stageButtons=[...document.querySelectorAll('[data-piper-stage]')];
  const piper=document.querySelector('[data-figure="piper"] svg');
  const piperBounds=piper.viewBox.baseVal;
  const stageWidth=piperBounds.width/4,stageHeight=piperBounds.height;
  const cropPiper=viewportClip(piper,'piper-stage-window');
  function showStage(index){
    cropPiper(index*stageWidth,0,stageWidth,stageHeight);
    piper.setAttribute('viewBox',`${index*stageWidth} 0 ${stageWidth} ${stageHeight}`);
    piper.setAttribute('aria-label',`PiPER ${stageButtons[index].textContent}: real and GS observations, head and external cameras`);
    stageButtons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    if(!reduced.matches)piper.animate([{opacity:.25},{opacity:1}],{duration:260,easing:'ease-out'});
  }
  stageButtons.forEach((b,i)=>b.addEventListener('click',()=>showStage(i)));
  document.querySelector('.stage-tabs').addEventListener('keydown',e=>{
    const i=stageButtons.indexOf(document.activeElement);if(i<0)return;
    let next;if(e.key==='ArrowRight')next=(i+1)%4;else if(e.key==='ArrowLeft')next=(i+3)%4;else return;
    e.preventDefault();stageButtons[next].focus();showStage(next);
  });
  showStage(0);
  // On phones show two adjacent Replica scenes at readable size.
  const replica=document.querySelector('[data-figure="replica"] svg'),replicaBox=[...replica.getAttribute('viewBox').split(' ').map(Number)];
  const cropReplica=viewportClip(replica,'replica-scene-window');
  replica.removeAttribute('width');replica.removeAttribute('height');
  const mobile=matchMedia('(max-width:700px)');let scene=0;
  const controls=document.createElement('div');controls.className='replica-tools';
  const previous=document.createElement('button'),next=document.createElement('button');
  previous.type=next.type='button';previous.textContent='Previous scenes';next.textContent='Next scenes';controls.append(previous,next);
  document.querySelector('.replica-figure .figure-view').after(controls);
  function showScenes(){cropReplica(mobile.matches?replicaBox[2]/7*scene:0,0,mobile.matches?replicaBox[2]/7*2:replicaBox[2],replicaBox[3]);replica.setAttribute('viewBox',mobile.matches?`${replicaBox[2]/7*scene} 0 ${replicaBox[2]/7*2} ${replicaBox[3]}`:replicaBox.join(' '));previous.disabled=scene===0;next.disabled=scene===5;}
  previous.addEventListener('click',()=>{scene=Math.max(0,scene-1);showScenes();});next.addEventListener('click',()=>{scene=Math.min(5,scene+1);showScenes();});mobile.addEventListener('change',showScenes);showScenes();
  document.getElementById('bib-copy').addEventListener('click',async()=>{
    const text=document.getElementById('bib-text').textContent;let copied=false;
    try{await navigator.clipboard.writeText(text);copied=true;}catch{const input=document.createElement('textarea');input.value=text;input.style.cssText='position:fixed;opacity:0';document.body.append(input);input.select();copied=document.execCommand('copy');input.remove();}
    document.getElementById('copy-status').textContent=copied?'Citation copied.':'Select the citation text to copy it.';
  });
})();
