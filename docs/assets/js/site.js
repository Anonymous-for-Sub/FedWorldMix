/* Page controls; scientific figures are the paper's original vector geometry. */
(() => {
  'use strict';
  document.getElementById('figure-colors').addEventListener('click',event=>{
    const paper=document.body.toggleAttribute('data-paper-colors');
    event.currentTarget.textContent=paper?'Dark colors':'Paper colors';
    event.currentTarget.setAttribute('aria-pressed',String(paper));
  });
  document.getElementById('bib-copy').addEventListener('click',async()=>{
    const text=document.getElementById('bib-text').textContent;
    let copied=false;
    try {await navigator.clipboard.writeText(text);copied=true;}catch{
      const input=document.createElement('textarea');input.value=text;
      input.style.cssText='position:fixed;opacity:0';document.body.append(input);
      input.select();copied=document.execCommand('copy');input.remove();
    }
    document.getElementById('copy-status').textContent=copied?'Citation copied.':'Select the citation text to copy it.';
  });
})();
