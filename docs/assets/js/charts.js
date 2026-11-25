/* Animate the paper's actual SVG marks. Axes, coordinates and final values stay
   unchanged; finished animations are removed to restore the exact source SVG. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const controllers=[];
  const series=['24445c','2a9d8f','e83e8c','a61e63','e9b949'];
  const color=(node,attr)=>(node.getAttribute(attr)||'').match(/--fig-([0-9a-f]{6})/)?.[1];
  function wrap(node,companions=[]){
    const group=document.createElementNS(NS,'g');node.parentNode.insertBefore(group,node);group.append(node);
    for(const other of companions)if(other.parentNode===group.parentNode)group.append(other);
    return group;
  }
  const fade=[{opacity:0},{opacity:1}];
  const grow=[{clipPath:'inset(0 100% 0 0)'},{clipPath:'inset(0 0% 0 0)'}];
  const pop=[{opacity:0,transform:'scale(.3)'},{opacity:1,transform:'scale(1)'}];
  function prepare(view){
    const key=view.dataset.figure,svg=view.querySelector('svg'),figure=view.closest('figure');
    const paths=[...svg.querySelectorAll('path')].filter(p=>!p.closest('defs'));
    const glyphs=[...svg.querySelectorAll('use[data-text]')];
    const specs=[];let markCount=0;
    function add(el,frames,delay=0,duration=650){specs.push({el,frames,delay,duration});}
    function pair(path){return paths.filter(p=>p!==path&&p.getAttribute('d')===path.getAttribute('d')&&p.getAttribute('transform')===path.getAttribute('transform')&&p.getAttribute('fill')==='none');}
    function mark(path,frames,delay,duration){
      const group=wrap(path,pair(path));group.style.transformBox='fill-box';group.style.transformOrigin='center';
      group.dataset.chartMark=key;markCount++;add(group,frames,delay,duration);return group;
    }
    const box=p=>p.getBBox();
    if(key==='retention'){
      const points=paths.filter(p=>{const b=box(p);return color(p,'fill')&&color(p,'fill')!=='ffffff'&&b.width>3&&b.width<10&&b.height>3&&b.height<10;});
      points.forEach((p,i)=>mark(p,pop,i*240,550));
      figure.dataset.animationKind='scatter-points';
    }else if(key==='rounds'||key==='training'){
      const lines=paths.filter(p=>{const b=box(p);return p.getAttribute('fill')==='none'&&series.includes(color(p,'stroke'))&&b.width>70&&b.height>10;});
      if(key==='training'){
        paths.filter(p=>{const b=box(p);return series.includes(color(p,'fill'))&&b.width>70&&b.height>10;}).forEach(p=>{
          const i=Math.max(0,lines.findIndex(line=>color(line,'stroke')===color(p,'fill')));mark(p,grow,i*220,1700);
        });
      }
      lines.forEach((p,i)=>mark(p,grow,i*220,1700));
      if(key==='rounds'){
        const rootInverse=svg.getCTM().inverse();
        const position=p=>{const b=box(p);return new DOMPoint(b.x+b.width/2,b.y+b.height/2).matrixTransform(p.getCTM()).matrixTransform(rootInverse);};
        const xs=new Set();for(const line of lines){const nums=line.getAttribute('d').match(/[-+]?(?:\d*\.)?\d+/g).map(Number);for(let i=0;i<nums.length;i+=2)xs.add(Math.round(nums[i]*10));}
        paths.filter(p=>{const b=box(p);return (series.includes(color(p,'fill'))||(color(p,'fill')==='ffffff'&&pair(p).some(q=>series.includes(color(q,'stroke')))))&&b.width>1&&b.width<8&&b.height>1&&b.height<8&&xs.has(Math.round(position(p).x*10));}).forEach(p=>{
          const order=Math.max(0,(position(p).x-41.3)/115.4);
          const ink=color(p,'fill')==='ffffff'?color(pair(p)[0],'stroke'):color(p,'fill');
          const seriesOrder=Math.max(0,lines.findIndex(line=>color(line,'stroke')===ink));
          mark(p,pop,order*1700+seriesOrder*220,300);
        });
      }
      figure.dataset.animationKind=key==='rounds'?'round-trajectories':'optimization-traces';
    }else if(key==='ablations'){
      const bars=paths.filter(p=>{const b=box(p);return ['e83e8c','f8c8dc'].includes(color(p,'fill'))&&b.width>.5&&b.height>4&&b.height<20;}).sort((a,b)=>b.getBBox().y-a.getBBox().y);
      bars.forEach((p,i)=>{
        const rect=p.getBoundingClientRect();mark(p,grow,i*170,950);
        glyphs.filter(g=>{const r=g.getBoundingClientRect();return r.x>=rect.x-2&&r.y+r.height/2>rect.y&&r.y+r.height/2<rect.bottom;}).forEach(g=>add(g,fade,i*170+800,250));
      });figure.dataset.animationKind='horizontal-bars';
    }else if(key==='fidelity'){
      const violins=paths.filter(p=>{const b=box(p);return color(p,'fill')==='e83e8c'&&b.width>20&&b.height>8;}).sort((a,b)=>b.getBBox().y-a.getBBox().y);
      violins.forEach((p,i)=>mark(p,[{clipPath:'inset(0 50% 0 50%)',opacity:.3},{clipPath:'inset(0 0% 0 0%)',opacity:1}],i*350,1000));
      paths.filter(p=>color(p,'stroke')==='a61e63').forEach(p=>add(p,fade,850,500));
      glyphs.filter(g=>color(g,'fill')==='24445c').forEach(g=>add(g,fade,1000,400));
      figure.dataset.animationKind='distribution-spread';
    }else if(key==='cameras'){
      const cells=paths.filter(p=>{const b=box(p);return color(p,'fill')&&color(p,'fill')!=='ffffff'&&b.width>12&&b.width<25&&b.height>15&&b.height<28;});
      cells.forEach((p,i)=>{const rect=p.getBoundingClientRect();mark(p,pop,i*65,500);
        glyphs.filter(g=>{const r=g.getBoundingClientRect();return r.x>=rect.x&&r.right<=rect.right+1&&r.y+r.height/2>rect.y&&r.y+r.height/2<rect.bottom;}).forEach(g=>add(g,fade,i*65+200,250));
      });figure.dataset.animationKind='camera-cells';
    }else{return null;}
    figure.dataset.animationMarks=String(markCount);
    let animations=[],played=false,inView=false,generation=0;
    function finish(){generation++;animations.forEach(a=>a.cancel());animations=[];figure.dataset.animationState='complete';}
    function play(){
      finish();played=true;if(reduced.matches){figure.dataset.animationState='reduced';return;}
      const run=++generation;figure.dataset.animationState='running';
      animations=specs.map(s=>s.el.animate(s.frames,{delay:s.delay,duration:s.duration,easing:'cubic-bezier(.2,.65,.3,1)',fill:'both'}));
      Promise.allSettled(animations.map(a=>a.finished)).then(()=>{if(run===generation)finish();});
    }
    const observer=new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;if(inView&&!played)play();},{threshold:.2});observer.observe(view);
    const control={play,finish,replay(){played=false;if(inView)play();else finish();}};controllers.push(control);return control;
  }
  document.querySelectorAll('[data-figure]').forEach(prepare);
  document.getElementById('replay-charts').addEventListener('click',()=>controllers.forEach(c=>c.replay()));
  reduced.addEventListener('change',()=>{if(reduced.matches)controllers.forEach(c=>c.finish());});
  // Quiet entrance for non-chart content; no hidden content when JS is absent.
  const entries=new IntersectionObserver(list=>list.forEach(entry=>{if(!entry.isIntersecting)return;entries.unobserve(entry.target);if(!reduced.matches)entry.target.animate([{opacity:.25,translate:'0 10px'},{opacity:1,translate:'0 0'}],{duration:550,easing:'ease-out'});}),{threshold:.12});
  document.querySelectorAll('.world-row figure,.quality-row>.paper-figure,.piper-grid figure,.resource-links a').forEach(e=>entries.observe(e));
})();
