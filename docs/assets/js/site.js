/* FedVista interactions. Results transcribed from the current manuscript. */
(() => {
  'use strict';
  const rows = [
    {name:'Frozen',pooled:36.11,appearance:36.50,viewpoint:38.17,mixed:33.67,worst:33.67,canonical:70.17},
    {name:'Real',pooled:41.33,appearance:37.33,viewpoint:43.67,mixed:43.00,worst:37.33,canonical:77.83},
    {name:'2D',pooled:42.22,appearance:41.83,viewpoint:42.33,mixed:42.50,worst:41.83,canonical:78.50},
    {name:'GS',pooled:40.56,appearance:38.00,viewpoint:43.83,mixed:39.83,worst:38.00,canonical:78.00},
    {name:'NaiveJoint',pooled:42.72,appearance:42.17,viewpoint:42.83,mixed:43.17,worst:42.17,canonical:77.17},
    {name:'VistaMix',pooled:46.00,appearance:42.83,viewpoint:47.00,mixed:48.17,worst:42.83,canonical:78.17}
  ];
  const info = {
    pooled:['Pooled OOD improvement','95% confidence interval [1.06, 5.50]. The larger gains under viewpoint and mixed shifts support geometric variation as a complement to appearance augmentation.'],
    appearance:['Appearance-shift improvement','2D augmentations provide appearance diversity. VistaMix retains their contribution while admitting valid GS corrections.'],
    viewpoint:['Viewpoint-shift improvement','Structured GS views provide geometric variation. VistaMix reaches 47.00% success under viewpoint shifts, compared with 42.83% for NaiveJoint.'],
    mixed:['Mixed-shift improvement','Combining camera and appearance changes is the most favorable comparison with NaiveJoint: 48.17% versus 43.17% success.'],
    canonical:['Canonical-domain improvement','VistaMix retains 78.17% canonical-domain success, compared with 77.17% for NaiveJoint, while improving robustness to visual shifts.']
  };
  function selectMetric(key) {
    document.querySelectorAll('[data-metric]').forEach(b => { const on=b.dataset.metric===key; b.setAttribute('aria-selected',String(on)); b.tabIndex=on?0:-1; });
    document.getElementById('result-panel').setAttribute('aria-labelledby','tab-'+key);
    const bars=document.getElementById('result-bars'); bars.replaceChildren();
    rows.forEach(row => {
      const el=document.createElement('div');el.className='result-row'+(row.name==='VistaMix'?' ours':'');
      const name=document.createElement('span');name.textContent=row.name;
      const track=document.createElement('span');track.className='bar-track';track.setAttribute('aria-hidden','true');
      const fill=document.createElement('i');fill.style.width=row[key]+'%';track.append(fill);
      const value=document.createElement('b');value.textContent=row[key].toFixed(2);el.append(name,track,value);bars.append(el);
    });
    const gain=(rows[5][key]-rows[4][key]).toFixed(2);
    const number=document.getElementById('gain-number');number.replaceChildren(document.createTextNode('+'+gain));
    const unit=document.createElement('span');unit.textContent='pp';number.append(unit);
    document.getElementById('gain-title').textContent=info[key][0];document.getElementById('gain-note').textContent=info[key][1];
  }
  document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>selectMetric(b.dataset.metric)));
  selectMetric('pooled');
  const table=document.getElementById('full-results');
  rows.forEach(row=>{const tr=document.createElement('tr');const name=document.createElement('th');name.scope='row';name.textContent=row.name;tr.append(name);['pooled','appearance','viewpoint','mixed','worst','canonical'].forEach(k=>{const td=document.createElement('td');td.textContent=row[k].toFixed(2);tr.append(td);});table.append(tr);});
  const worlds={
    piper:{file:'02_piper_real_and_gs',width:1280,height:720,title:'PiPER: real observations and GS renders',description:'Synchronized head and external camera observations show the real PiPER trajectory alongside its Gaussian Splatting representation.',timing:'Recorded source comparison · original playback timing'},
    robotwin:{file:'03_robotwin_truth_and_gs',width:1280,height:720,title:'RoboTwin: reference and component-3DGS views',description:'Near and far cameras reveal corresponding reference and component-3DGS observations of a RoboTwin trajectory.',timing:'Recorded source comparison · original playback timing'},
    replica:{file:'04_replica8_go2_and_g1',width:1280,height:360,title:'Replica-8: Go2 and G1 across eight scenes',description:'Go2 and G1 states are rendered across all eight Replica reconstructions. SUCCESS labels describe the source controller episodes in this montage.',timing:'Source coverage montage · each scene normalized to 6 seconds'},
    robosplat:{file:'05_robosplat_visual_worlds',width:1280,height:720,title:'RoboSplat: four visual conditions',description:'The same RoboSplat demonstration appears under canonical, camera, lighting and combined visual conditions, illustrating complementary forms of visual variation.',timing:'Recorded visual-source comparison · original playback timing'}
  };
  document.querySelectorAll('[data-world]').forEach(b=>b.addEventListener('click',()=>{
    const key=b.dataset.world,w=worlds[key];document.querySelectorAll('[data-world]').forEach(x=>{const on=x===b;x.setAttribute('aria-selected',String(on));x.tabIndex=on?0:-1;});
    document.getElementById('world-panel').setAttribute('aria-labelledby','world-'+key);
    const video=document.getElementById('world-video');video.pause();video.width=w.width;video.height=w.height;video.style.aspectRatio=w.width+' / '+w.height;video.poster='assets/media/'+w.file+'.jpg';video.querySelector('source').src='assets/media/'+w.file+'.mp4';video.setAttribute('aria-label',w.title);video.load();
    document.getElementById('world-title').textContent=w.title;document.getElementById('world-description').textContent=w.description;document.getElementById('world-timing').textContent=w.timing;
  }));
  document.querySelectorAll('[role=tablist]:not(.overview-steps)').forEach(list=>list.addEventListener('keydown',e=>{
    const keys=['ArrowRight','ArrowDown','ArrowLeft','ArrowUp','Home','End'];if(!keys.includes(e.key))return;
    const buttons=[...list.querySelectorAll('[role=tab]')];const current=buttons.indexOf(document.activeElement);if(current<0)return;e.preventDefault();
    const next=e.key==='Home'?0:e.key==='End'?buttons.length-1:(current+(['ArrowRight','ArrowDown'].includes(e.key)?1:-1)+buttons.length)%buttons.length;
    buttons[next].focus();buttons[next].click();
  }));
  document.getElementById('bib-copy').addEventListener('click',async()=>{
    const text=document.getElementById('bib-text').textContent;let copied=false;
    try{await navigator.clipboard.writeText(text);copied=true;}catch{
      const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();copied=document.execCommand('copy');area.remove();
    }
    document.getElementById('copy-status').textContent=copied?'Citation copied.':'Select the citation text to copy it.';
  });
  const navLinks=[...document.querySelectorAll('.nav-links a')];
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)navLinks.forEach(a=>a.classList.toggle('is-active',a.hash==='#'+entry.target.id));}),{rootMargin:'-20% 0px -50% 0px'});
  ['method','results','worlds','demos','resources'].forEach(id=>observer.observe(document.getElementById(id)));
  document.querySelectorAll('video').forEach(video=>video.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();})));
})();
