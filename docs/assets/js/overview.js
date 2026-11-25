/* FedVista native SVG overview. No model inference runs in this diagram. */
(() => {
  'use strict';
  const host=document.getElementById('overview-canvas'), overview=document.getElementById('overview');
  if(!host||!overview)return;
  const mobile=matchMedia('(max-width: 680px)'), reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const descriptions=[
    ['01 / Local sources','Real observations, 2D augmentations and eligible GS views share the source action target. GS admission uses local coverage, validity and alignment checks.'],
    ['02 / VLA adapter','Each source passes through the shared policy. Vision-encoder and connector LoRA parameters are trainable; the action expert stays frozen. Per-source action losses supply the gradients.'],
    ['03 / VistaMix','Real + 2D forms the base. The GS gradient is restricted to perception parameters, projected for severe conflict, weighted by validity, and capped relative to the base norm.'],
    ['04 / FedAvg','The local optimizer produces an adapter update. Clients send that update and their sample count; the server applies sample-weighted FedAvg and returns the global adapter for the next round.']
  ];
  const style=`
    .ov-bg{fill:#fff}.ov-scope{fill:#fbfcfd;stroke:#cbd5df;stroke-dasharray:5 5}.ov-stage-frame{fill:#fff;stroke:#cad4df;stroke-width:1.3;transition:stroke .25s}.ov-stage.active .ov-stage-frame{stroke:#b30064;stroke-width:2}.ov-box{fill:#edf4fb;stroke:#b3c9dd;stroke-width:1}.ov-real{fill:#fbeef0;stroke:#dec2ca}.ov-gs{fill:#fff4df;stroke:#dec59a}.ov-mix{fill:#fbf0f6;stroke:#d7b1c4}.ov-server{fill:#eef4f9;stroke:#bacddc}.ov-text{font-family:Arial,Helvetica,sans-serif;fill:#20374b;font-size:14px}.ov-small{font-size:11px;fill:#586c7c}.ov-label{font-size:10px;fill:#596c7d;letter-spacing:1px}.ov-heading{font-size:14px;font-weight:600}.ov-flow{fill:none;stroke:#97aaba;stroke-width:1.5;stroke-linejoin:round}.ov-return{stroke-dasharray:5 5;stroke:#6c87a0}.ov-flow.active{stroke:#b30064;stroke-width:2}.ov-running .ov-flow.active{stroke-dasharray:7 8;animation:ov-travel 1.1s linear infinite}.ov-stage{cursor:pointer}.ov-stage:hover .ov-stage-frame{stroke:#567c9d}@keyframes ov-travel{to{stroke-dashoffset:-30}}@media(prefers-reduced-motion:reduce){.ov-running .ov-flow.active{animation:none;stroke-dasharray:none}}
  `;
  const rect=(x,y,w,h,cls='ov-box',r=9)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`;
  const txt=(x,y,str,cls='',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" class="ov-text ${cls}">${str}</text>`;
  const flow=(d,stage,ret=false)=>`<path d="${d}" class="ov-flow${ret?' ov-return':''}" data-flow="${stage}" marker-end="url(#ov-arrow)"/>`;
  const group=(stage,content)=>`<g class="ov-stage" data-stage="${stage}">${content}</g>`;
  function desktop(){return `
    ${rect(10,48,866,320,'ov-scope',12)}${txt(28,32,'CLIENT i · local observations and training','ov-label','start')}
    ${group(0,`${rect(28,86,162,66,'ov-stage-frame ov-real')}${txt(109,113,'Real observations','ov-heading')}${txt(109,136,'Original action targets','ov-small')}${rect(28,175,162,66,'ov-stage-frame')}${txt(109,202,'2D augmentation','ov-heading')}${txt(109,225,'Appearance variation','ov-small')}${rect(28,264,162,80,'ov-stage-frame ov-gs')}${txt(109,291,'GS views','ov-heading')}${txt(109,312,'Coverage + alignment','ov-small')}${txt(109,330,'Validity gate','ov-small')}`)}
    ${flow('M190 119 H220 V208 H260',1)}${flow('M190 208 H260',1)}${flow('M190 304 H235 V235 H260',1)}
    ${group(1,`${rect(260,86,230,258,'ov-stage-frame')}${txt(375,110,'SHARED VLA','ov-label')}${rect(280,135,190,67)}${txt(375,161,'Vision encoder','ov-heading')}${txt(375,185,'+ connector · LoRA','ov-small')}${rect(280,231,190,60,'ov-box')}${txt(375,256,'Action expert','ov-heading')}${txt(375,277,'Frozen parameters','ov-small')}${flow('M375 202 V231',1)}${txt(375,322,'Per-source action losses','ov-small')}`)}
    ${flow('M490 208 H538',2)}
    ${group(2,`${rect(538,86,304,258,'ov-stage-frame')}${txt(690,110,'VISTAMIX','ov-label')}${rect(560,132,260,56)}${txt(690,154,'Base gradient','ov-small')}${txt(690,176,'Real + 0.75 × 2D','ov-heading')}${rect(560,214,260,69,'ov-box ov-gs')}${txt(690,237,'GS correction','ov-heading')}${txt(690,258,'Restrict → project → weight → cap','ov-small')}${txt(690,275,'0.25 × mean validity × projected GS','ov-small')}${rect(575,305,245,27,'ov-box ov-mix',6)}${txt(697,324,'Base + bounded GS correction','ov-small')}${flow('M560 160 H549 V318 H575',2)}${flow('M697 283 V305',2)}`)}
    ${flow('M842 318 H883 V166 H914',3)}${txt(881,141,'Δᵢ, nᵢ','ov-small')}
    ${group(3,`${rect(914,86,226,258,'ov-stage-frame ov-server')}${txt(1027,110,'FEDERATED SERVER','ov-label')}${rect(934,136,186,60)}${txt(1027,160,'Client adapter update','ov-heading')}${txt(1027,182,'and sample count','ov-small')}${rect(934,224,186,65)}${txt(1027,250,'Sample-weighted','ov-heading')}${txt(1027,273,'FedAvg','ov-heading')}${flow('M1027 196 V224',3)}${txt(1027,322,'Global adapter','ov-small')}`)}
    ${flow('M1027 344 V407 H375 V344',3,true)}${txt(701,433,'Global adapter returned for the next round','ov-small')}
  `;}
  function phone(){return `
    ${txt(15,23,'CLIENT i · local observations and training','ov-label','start')}
    ${group(0,`${rect(12,45,104,76,'ov-stage-frame ov-real')}${txt(64,72,'Real','ov-heading')}${txt(64,94,'Action targets','ov-small')}${rect(128,45,104,76,'ov-stage-frame')}${txt(180,72,'2D','ov-heading')}${txt(180,94,'Appearance','ov-small')}${rect(244,45,104,76,'ov-stage-frame ov-gs')}${txt(296,72,'GS','ov-heading')}${txt(296,94,'Validity gate','ov-small')}`)}
    ${flow('M64 121 V145 H180 V172',1)}${flow('M180 121 V172',1)}${flow('M296 121 V145 H180',1)}
    ${group(1,`${rect(20,172,320,162,'ov-stage-frame')}${txt(180,199,'SHARED VLA','ov-label')}${rect(36,218,141,66)}${txt(106,243,'Vision + connector','ov-small')}${txt(106,266,'Trainable LoRA','ov-small')}${rect(198,218,126,66)}${txt(261,243,'Action expert','ov-small')}${txt(261,266,'Frozen','ov-small')}${flow('M177 250 H198',1)}${txt(180,314,'Per-source action losses','ov-small')}`)}
    ${flow('M180 334 V374',2)}
    ${group(2,`${rect(20,374,320,246,'ov-stage-frame')}${txt(180,401,'VISTAMIX','ov-label')}${rect(42,420,278,53)}${txt(181,441,'Base gradient','ov-small')}${txt(181,461,'Real + 0.75 × 2D','ov-heading')}${rect(42,496,278,69,'ov-box ov-gs')}${txt(181,520,'GS correction','ov-heading')}${txt(181,540,'Restrict → project → weight → cap','ov-small')}${txt(181,557,'0.25 × mean validity × projected GS','ov-small')}${rect(54,587,266,25,'ov-box ov-mix',5)}${txt(187,604,'Base + bounded GS correction','ov-small')}${flow('M42 447 H31 V600 H54',2)}${flow('M187 565 V587',2)}`)}
    ${flow('M180 620 V662',3)}${txt(208,646,'Δᵢ, nᵢ','ov-small','start')}
    ${group(3,`${rect(20,662,320,173,'ov-stage-frame ov-server')}${txt(180,690,'FEDERATED SERVER','ov-label')}${rect(36,708,132,66)}${txt(102,733,'Adapter update','ov-small')}${txt(102,755,'+ sample count','ov-small')}${rect(190,708,134,66)}${txt(257,733,'Sample-weighted','ov-small')}${txt(257,755,'FedAvg','ov-heading')}${flow('M168 741 H190',3)}${txt(180,813,'Global adapter','ov-heading')}`)}
    ${flow('M180 835 V857 H7 V350 H65 V334',3,true)}${txt(180,884,'Global adapter → next round','ov-small')}
  `;}
  let stage=0,timer=null,inView=false,playing=!reduced.matches;
  function render(){
    host.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${mobile.matches?'360 904':'1160 452'}" role="img" aria-labelledby="overview-svg-title overview-svg-desc"><title id="overview-svg-title">FedVista overview</title><desc id="overview-svg-desc">Real, 2D and valid GS observations pass through a shared VLA with perception LoRA. VistaMix protects the Real-plus-2D base gradient. Clients upload adapter updates for sample-weighted FedAvg, then receive the global adapter.</desc><style>${style}</style><defs><marker id="ov-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="#7d92a5"/></marker></defs>${mobile.matches?phone():desktop()}</svg>`;
    host.querySelectorAll('[data-stage]').forEach(g=>g.addEventListener('click',()=>{playing=false;setStage(Number(g.dataset.stage));sync();}));setStage(stage);sync();
  }
  function setStage(value){
    stage=value;host.querySelectorAll('[data-stage]').forEach(g=>g.classList.toggle('active',Number(g.dataset.stage)===stage));host.querySelectorAll('[data-flow]').forEach(p=>p.classList.toggle('active',Number(p.dataset.flow)===stage));
    overview.querySelectorAll('[data-overview-step]').forEach(b=>{const on=Number(b.dataset.overviewStep)===stage;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});
    document.getElementById('overview-description').setAttribute('aria-labelledby','overview-tab-'+stage);document.getElementById('overview-step-label').textContent=descriptions[stage][0];document.getElementById('overview-step-text').textContent=descriptions[stage][1];
  }
  function sync(){
    if(timer){clearInterval(timer);timer=null;}
    const running=playing&&inView&&!document.hidden;
    overview.dataset.playing=String(playing);host.querySelector('svg')?.classList.toggle('ov-running',running&&!reduced.matches);
    const button=document.getElementById('overview-play');button.textContent=playing?'Pause':'Play';button.setAttribute('aria-pressed',String(playing));
    if(running)timer=setInterval(()=>setStage((stage+1)%4),4000);
  }
  overview.querySelectorAll('[data-overview-step]').forEach(b=>b.addEventListener('click',()=>{playing=false;setStage(Number(b.dataset.overviewStep));sync();}));
  overview.querySelector('.overview-steps').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?3:(stage+(e.key==='ArrowRight'?1:3))%4;document.getElementById('overview-tab-'+next).focus();document.getElementById('overview-tab-'+next).click();});
  document.getElementById('overview-play').addEventListener('click',()=>{playing=!playing;sync();});
  new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;sync();},{threshold:.15}).observe(overview);
  document.addEventListener('visibilitychange',sync);mobile.addEventListener('change',render);reduced.addEventListener('change',()=>{if(reduced.matches)playing=false;sync();});render();
})();
