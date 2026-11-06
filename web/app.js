'use strict';
const $=id=>document.getElementById(id);
let ready=false,busy=false,revision=0,controller=null,examples=[];
function update(){const s=$('text').value.trim();$('count').textContent=(s?s.split(/\s+/).length:0)+' palavras';$('analyze').disabled=!ready||busy||!s;}
function reset(){revision++;if(controller)controller.abort();busy=false;$('output').hidden=true;$('loading').hidden=true;$('empty').hidden=false;$('error').hidden=true;$('example-note').textContent='';$('result-panel').setAttribute('aria-busy','false');update();}
$('text').addEventListener('input',reset);
$('clear').addEventListener('click',()=>{$('text').value='';reset();$('text').focus();});
for(const [id,label] of [['human',0],['ai',1]])$(id).addEventListener('click',()=>{const ex=examples.find(x=>x.label===label);if(!ex)return;$('text').value=ex.text;reset();$('example-note').textContent='Linha '+ex.row_id+' da base dos autores, conjunto de validação. Rótulo da base: '+(label?'IA':'humano')+'. Exemplo não escolhido pelo acerto do modelo.';});
$('analyze').addEventListener('click',async()=>{
 if(!ready||busy||!$('text').value.trim())return;
 busy=true;update();const ticket=revision;controller=new AbortController();
 $('empty').hidden=true;$('output').hidden=true;$('loading').hidden=false;$('error').hidden=true;$('result-panel').setAttribute('aria-busy','true');
 try{
  const res=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-Local-UI':'1'},body:JSON.stringify({text:$('text').value}),signal:controller.signal});
  const r=await res.json();if(ticket!==revision)return;if(!res.ok)throw new Error(r.error||'Não foi possível analisar.');
  $('prediction').textContent=r.prediction==='IA'?'Padrão de IA':'Padrão humano';$('score').textContent=(r.score_ai*100).toFixed(1)+'%';$('bar').style.width=(r.score_ai*100)+'%';
  $('details').textContent=r.tokens_analyzed+' tokens analisados · '+(r.unknown_fraction*100).toFixed(1)+'% fora do vocabulário';
  $('truncation').hidden=!r.truncated;$('truncation').textContent='Texto longo: somente os primeiros 200 tokens após limpeza foram analisados.';
  $('output').hidden=false;
 }catch(e){if(ticket===revision&&e.name!=='AbortError'){$('error').textContent=e instanceof TypeError?'Conexão indisponível. Mantenha o servidor aberto.':e.message;$('error').hidden=false;$('empty').hidden=false;}}
 finally{if(ticket===revision){busy=false;$('loading').hidden=true;$('result-panel').setAttribute('aria-busy','false');update();}}
});
fetch('/api/info').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(info=>{ready=true;examples=info.examples;$('accuracy').textContent=(info.report.test.accuracy*100).toFixed(2)+'%';$('human').disabled=!examples.some(x=>x.label===0);$('ai').disabled=!examples.some(x=>x.label===1);update();}).catch(()=>{$('error').textContent='Não foi possível carregar o experimento. Reinicie o servidor local.';$('error').hidden=false;});
