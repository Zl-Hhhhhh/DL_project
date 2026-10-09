'use strict';
const $ = id => document.getElementById(id);
const state = {config:null, file:null, image:null, objectURL:null, style:null, composition:'center', busy:false, result:null, allowTruncated:false, detection:null, view:'source'};
const compositionNames = {left:'靠左', center:'居中', right:'靠右'};
Object.assign(state,{mode:'collage',styleImage:null,styleURL:null,materials:[],materialURLs:[],materialSource:'builtin',size:1024});
Object.assign(state,{effect:'existing',posterPosition:'center'});
const posterNames={center:'居中',top_left:'左上三分',top_right:'右上三分',bottom_left:'左下三分',bottom_right:'右下三分'};
function updateEffectControls(){
  const collage=state.mode==='collage';
  $('collage-effects').hidden=!collage;
  $('collage-styles').hidden=!collage || state.effect!=='existing';
  document.querySelector('.composition-section').hidden=!collage || state.effect!=='existing';
  $('color-focus-controls').hidden=state.effect!=='color_focus';
  $('poster-controls').hidden=state.effect!=='poster';
  $('effect-note').textContent={existing:'移动主体，只填色遮盖主体原来的位置。',color_focus:'保留原有构图，通过背景调色突出主体。',poster:'保留主体原色，放在米白底上；原背景会移除。'}[state.effect];
}
function ready(){return !!(state.image && state.config && (state.mode==='collage' || state.styleImage) && (!$('enable-collagerl').checked || (state.materialSource==='builtin'?state.config.builtin_materials?.count:state.materials.length)));}
function error(message, retry=false) { $('error-message').textContent=message; $('error-box').hidden=false; $('truncated-retry').hidden=!retry; }
function clearError() { $('error-box').hidden=true; $('truncated-retry').hidden=true; }
function setBusy(busy) {
  state.busy=busy; document.querySelector('.controls').classList.toggle('busy',busy);
  document.querySelectorAll('.controls button').forEach(button=>button.disabled=busy);
  document.querySelectorAll('.controls input').forEach(input=>input.disabled=busy);
  $('generate-button').disabled=busy || !ready();
  $('generate-label').textContent=busy?'正在生成':(state.mode==='collage'?'开始拼接':'开始风格化');
  document.querySelector('.generate-arrow').hidden=busy; $('button-spinner').hidden=!busy;
  $('generating-overlay').hidden=!busy; $('artboard').setAttribute('aria-busy',String(busy));
  if(busy){$('result-status').textContent='正在处理，请稍候'; $('download-button').hidden=true; $('view-toggle').hidden=true;}
  else if(state.result){$('download-button').hidden=false;$('view-toggle').hidden=false;}
}
function updatePreview() {
  const source=state.detection?.source || state.result?.source || state.objectURL;
  const url=state.view==='result' && state.result ? state.result.output : source;
  if(url){$('preview-image').src=url;$('preview-image').hidden=false;$('canvas-empty').hidden=true;}
  $('view-original').setAttribute('aria-pressed',String(state.view==='source'));
  $('view-result').setAttribute('aria-pressed',String(state.view==='result'));
  const overlay=$('detection-boxes');overlay.replaceChildren();
  const show=$('show-boxes').checked && state.view==='source' && !!state.detection;
  overlay.toggleAttribute('hidden',!show);
  if(!show)return;
  const threshold=Number($('confidence-threshold').value)/100;
  for(const box of state.detection.detections){
    if(box.confidence<=threshold)continue;
    const [x0,y0,x1,y1]=box.bbox_xyxy.map(value=>Math.max(0,Math.min(state.size,value)));
    const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    for(const [key,value] of Object.entries({x:x0,y:y0,width:x1-x0,height:y1-y0}))rect.setAttribute(key,value);
    overlay.append(rect);
  }
}
function invalidateResult() {
  state.result=null; $('download-button').hidden=true; $('view-toggle').hidden=true; clearError();
  $('preview-title').textContent='画布预览';
  $('preview-description').textContent=state.image?(state.mode==='stylize'?'上传风格图，调整强度后开始风格化':'选好样式，点击开始拼接'):'上传一张照片，开始创作';
  $('result-status').textContent=state.image?'已准备好':'等待上传';
  state.view='source';updatePreview();
  if(!state.busy)setBusy(false);
}
function groupKeyboard(event) {
  if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
  const buttons=[...event.currentTarget.querySelectorAll('[role=radio]')];
  const index=buttons.indexOf(document.activeElement); if(index<0)return;
  event.preventDefault();
  let next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(['ArrowLeft','ArrowUp'].includes(event.key)?-1:1)+buttons.length)%buttons.length;
  buttons[next].focus(); buttons[next].click();
}
async function loadFile(file) {
  if(state.busy || !file)return;
  clearError();
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){error('请选择 JPG、PNG 或 WebP 图片。');return;}
  if(file.size>(state.config?.max_bytes || 12*1024*1024)){error('图片过大，请使用 12 MB 以内的图片。');return;}
  setBusy(true); $('generation-stage').textContent='正在读取图片';
  let url;
  try{
    url=URL.createObjectURL(file); const image=new Image(); image.src=url; await image.decode();
    if(image.naturalWidth*image.naturalHeight>25000000)throw new Error('图片尺寸过大，请缩小到 2500 万像素以内。');
    const base64=await new Promise((resolve,reject)=>{const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]); reader.onerror=()=>reject(new Error('无法读取图片，请重新选择。')); reader.readAsDataURL(file);});
    if(state.objectURL)URL.revokeObjectURL(state.objectURL);
    state.objectURL=url; state.file=file; state.image=base64; state.allowTruncated=false; state.detection=null;
    $('input-thumb').src=url; $('file-name').textContent=file.name;
    $('file-info').textContent=`${image.naturalWidth} × ${image.naturalHeight} · ${(file.size/1024/1024).toFixed(1)} MB`;
    $('upload-empty').hidden=true; $('upload-selected').hidden=false;
    invalidateResult();
  }catch(err){if(url && url!==state.objectURL)URL.revokeObjectURL(url);error(err.message || '这张图片无法读取，请换一张试试。');}
  finally{setBusy(false);}
  if(url===state.objectURL && $('show-boxes').checked)await detectBoxes();
}
async function getJSON(url, options={}) {
  const response=await fetch(url,options); const value=await response.json();
  if(!response.ok)throw new Error(value.message || '请求未完成，请再试一次。');
  return value;
}
async function detectBoxes() {
  if(state.busy || !state.image || !state.config)return;
  state.view='source';updatePreview();
  if(state.detection)return;
  clearError();setBusy(true);$('generate-label').textContent='正在识别';$('generation-stage').textContent='正在识别图片';
  try{
    const submitted=await getJSON('/api/detect',{method:'POST',headers:{'Content-Type':'application/json','X-Collage-Token':state.config.token},body:JSON.stringify({image:state.image,style:state.style,composition:state.composition,size:state.size})});
    const started=Date.now();let job;
    while(true){
      job=await getJSON(`/api/jobs/${submitted.id}`);
      if(job.state!=='working')break;
      $('generation-stage').textContent=job.stage;
      if(Date.now()-started>300000)throw new Error('识别时间较长，请稍后重试。');
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    if(job.state!=='succeeded')throw new Error(job.message || '识别未完成，请重试。');
    const image=new Image();image.src=job.source;await image.decode();
    state.detection=job;updatePreview();$('result-status').textContent=state.result?'生成完成':'已准备好';
  }catch(err){error(err.message);$('result-status').textContent='识别未完成';}
  finally{setBusy(false);}
}
async function generate(allowTruncated=false) {
  if(state.busy || !ready())return;
  clearError(); setBusy(true); $('generation-stage').textContent='正在准备图片';
  const selection={style:state.mode==='collage' && state.effect!=='existing'?'original':state.style,composition:state.composition,mode:state.mode,size:state.size,postprocess:$('enable-collagerl').checked,
    collage_effect:state.effect,background_saturation:Number($('background-saturation').value)/100,poster_position:state.posterPosition,subject_size:Number($('subject-size').value)/100};
  try{
    const submitted=await getJSON('/api/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Collage-Token':state.config.token},body:JSON.stringify({image:state.image,...selection,allow_truncated:allowTruncated,style_image:state.mode==='stylize'?state.styleImage:undefined,prompt:$('style-prompt').value,style_strength:Number($('style-strength').value)/100,structure_strength:Number($('structure-strength').value)/100,material_source:state.materialSource,materials:selection.postprocess && state.materialSource==='custom'?state.materials:[]})});
    let job; const started=Date.now();
    while(true){
      job=await getJSON(`/api/jobs/${submitted.id}`);
      if(job.state!=='working')break;
      $('generation-stage').textContent=job.stage; $('result-status').textContent=job.stage;
      if(Date.now()-started>3600000)throw new Error('处理时间较长，请检查启动窗口。');
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
    if(job.state==='failed'){
      error(job.message,job.code==='subject_truncated'); $('result-status').textContent='未生成，请查看提示'; return;
    }
    const preview=new Image(); preview.src=job.output; await preview.decode();
    state.result={...job,...selection}; $('preview-image').src=job.output; $('preview-image').hidden=false; $('canvas-empty').hidden=true;
    $('preview-title').textContent='生成结果';
    const description=selection.collage_effect==='color_focus'?`主体保色 · 背景饱和度 ${Math.round(selection.background_saturation*100)}%`:selection.collage_effect==='poster'?`纯色拼贴 · ${posterNames[selection.poster_position]} · 主体 ${Math.round(selection.subject_size*100)}%`:`拼接 · ${state.config.styles.find(s=>s.id===selection.style).name} · ${compositionNames[selection.composition]}`;
    $('preview-description').textContent=(selection.mode==='stylize'?'风格化':description)+(selection.postprocess?' · CollageRL':'');
    $('view-original').setAttribute('aria-pressed','false'); $('view-result').setAttribute('aria-pressed','true');
    $('view-toggle').hidden=false;
    $('download-button').href=job.download; $('download-button').hidden=false;
    $('result-status').textContent=job.truncated?'已生成，使用了边缘主体':'生成完成';
    state.view='result';updatePreview();
    if(window.innerWidth<=700)document.querySelector('.preview-area').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  }catch(err){error(err.message || '无法连接本地服务，请确认启动窗口仍在运行。');$('result-status').textContent='生成未完成';}
  finally{setBusy(false);}
}
$('upload-button').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',event=>{loadFile(event.target.files[0]);event.target.value='';});
['dragenter','dragover'].forEach(type=>$('upload-button').addEventListener(type,event=>{event.preventDefault();if(!state.busy)$('upload-button').classList.add('dragging');}));
['dragleave','drop'].forEach(type=>$('upload-button').addEventListener(type,event=>{event.preventDefault();$('upload-button').classList.remove('dragging');if(type==='drop')loadFile(event.dataTransfer.files[0]);}));
window.addEventListener('dragover',event=>event.preventDefault());window.addEventListener('drop',event=>event.preventDefault());
$('sample-button').addEventListener('click',async()=>{try{const response=await fetch('/assets/example.jpg');if(!response.ok)throw new Error('示例图暂时无法读取，请上传自己的照片。');await loadFile(new File([await response.blob()],'示例图片.jpg',{type:'image/jpeg'}));}catch(err){error(err.message);}});
$('generate-button').addEventListener('click',()=>generate(false));
$('truncated-retry').addEventListener('click',()=>generate(true));
document.querySelectorAll('[data-composition]').forEach(button=>button.addEventListener('click',()=>{if(state.busy)return;state.composition=button.dataset.composition;document.querySelectorAll('[data-composition]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});invalidateResult();}));
document.querySelector('.composition-control').addEventListener('keydown',groupKeyboard);$('styles').addEventListener('keydown',groupKeyboard);
document.querySelectorAll('[data-composition]').forEach(b=>b.tabIndex=b.getAttribute('aria-checked')==='true'?0:-1);
$('view-original').addEventListener('click',()=>{if(!state.result)return;state.view='source';updatePreview();});
$('view-result').addEventListener('click',()=>{if(!state.result)return;state.view='result';updatePreview();});
$('show-boxes').addEventListener('change',()=>{$('threshold-control').hidden=!$('show-boxes').checked;if($('show-boxes').checked)detectBoxes();else updatePreview();});
$('confidence-threshold').addEventListener('input',()=>{$('threshold-value').textContent=`${$('confidence-threshold').value}%`;updatePreview();});
async function init(){try{
  state.config=await getJSON('/api/config');state.style=state.config.styles[0].id;
  if(!state.config.collage_effects?.includes('poster') || !state.config.conversion_modes?.includes('stylize') || !state.config.material_sources?.includes('builtin') || !state.config.output_sizes?.includes(512)){state.config=null;error('服务版本需要更新，请重新启动 start-ui.cmd 后刷新页面。');return;}
  const library=state.config.builtin_materials;
  $('builtin-materials-note').textContent=library.count?`报纸与杂志 · ${library.count} 张，已预选 ${library.used_count} 张用于生成`:'内置素材尚未安装，请选择自定义上传。';
  if(library.count){$('builtin-materials-preview').src=library.preview;$('builtin-materials-preview').hidden=false;}
  updateSize();
  state.config.styles.forEach((style,index)=>{
    const button=document.createElement('button');button.type='button';button.className='style-option';button.setAttribute('role','radio');button.setAttribute('aria-checked',String(index===0));button.tabIndex=index===0?0:-1;
    if(style.id==='original')button.classList.add('original-option');
    const img=document.createElement('img');img.src=style.preview;img.alt='';img.width=128;img.height=128;const label=document.createElement('strong');label.textContent=style.name;button.append(img,label);
    button.addEventListener('click',()=>{if(state.busy)return;state.style=style.id;[...$('styles').children].forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});invalidateResult();});$('styles').append(button);
  });setBusy(false);
}catch(err){error('无法连接本地服务，请确认启动窗口仍在运行，然后刷新页面。');}}
init();

document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{
  if(state.busy)return;state.mode=button.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});
  updateEffectControls();
  $('stylize-controls').hidden=state.mode!=='stylize';invalidateResult();
}));
document.querySelector('.mode-control').addEventListener('keydown',groupKeyboard);
document.querySelectorAll('[data-effect]').forEach(button=>button.addEventListener('click',()=>{
  if(state.busy)return;state.effect=button.dataset.effect;
  document.querySelectorAll('[data-effect]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});
  updateEffectControls();invalidateResult();
}));
document.querySelectorAll('[data-poster-position]').forEach(button=>button.addEventListener('click',()=>{
  if(state.busy)return;state.posterPosition=button.dataset.posterPosition;
  document.querySelectorAll('[data-poster-position]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});invalidateResult();
}));
document.querySelector('.effect-control').addEventListener('keydown',groupKeyboard);
document.querySelector('.poster-position-control').addEventListener('keydown',groupKeyboard);
['background-saturation','subject-size'].forEach(id=>$(id).addEventListener('input',()=>{$(id+'-value').textContent=$(id).value+'%';invalidateResult();}));
function updateSize(){
  $('canvas-dimensions').textContent=`${state.size} × ${state.size}`;
  $('preview-image').width=state.size;$('preview-image').height=state.size;
  $('detection-boxes').setAttribute('viewBox',`0 0 ${state.size} ${state.size}`);
}
document.querySelectorAll('[data-size]').forEach(button=>button.addEventListener('click',()=>{
  if(state.busy || state.size===Number(button.dataset.size))return;
  state.size=Number(button.dataset.size);state.detection=null;
  document.querySelectorAll('[data-size]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});
  updateSize();invalidateResult();if($('show-boxes').checked)detectBoxes();
}));
document.querySelector('.size-control').addEventListener('keydown',groupKeyboard);
async function readExtra(file,limit){
  if(!['image/jpeg','image/png','image/webp'].includes(file.type) || file.size>limit)throw new Error('请检查图片格式和文件大小。');
  const url=URL.createObjectURL(file);
  try{const img=new Image();img.src=url;await img.decode();if(img.naturalWidth*img.naturalHeight>25000000)throw new Error('图片不能超过 2500 万像素。');
    const encoded=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=()=>reject(new Error('无法读取图片'));r.readAsDataURL(file);});return {url,encoded};
  }catch(err){URL.revokeObjectURL(url);throw err;}
}
async function loadStyle(file){
  if(state.busy || !file)return;setBusy(true);$('generation-stage').textContent='正在读取风格图';
  try{const item=await readExtra(file,12*1024*1024);if(state.styleURL)URL.revokeObjectURL(state.styleURL);state.styleURL=item.url;state.styleImage=item.encoded;$('style-thumb').src=item.url;$('style-name').textContent=file.name;$('style-empty').hidden=true;$('style-selected').hidden=false;invalidateResult();}catch(err){error(err.message);}finally{setBusy(false);}
}
$('style-upload-button').addEventListener('click',()=>$('style-file-input').click());
$('style-file-input').addEventListener('change',e=>{loadStyle(e.target.files[0]);e.target.value='';});
['dragenter','dragover'].forEach(type=>$('style-upload-button').addEventListener(type,e=>{e.preventDefault();if(!state.busy)$('style-upload-button').classList.add('dragging');}));
['dragleave','drop'].forEach(type=>$('style-upload-button').addEventListener(type,e=>{e.preventDefault();$('style-upload-button').classList.remove('dragging');if(type==='drop')loadStyle(e.dataTransfer.files[0]);}));
['style-strength','structure-strength'].forEach(id=>$(id).addEventListener('input',()=>{$(id+'-value').textContent=$(id).value+'%';invalidateResult();}));
$('style-prompt').addEventListener('input',invalidateResult);
$('enable-collagerl').addEventListener('change',()=>{$('materials-panel').hidden=!$('enable-collagerl').checked;invalidateResult();});
document.querySelectorAll('[data-material-source]').forEach(button=>button.addEventListener('click',()=>{
  if(state.busy)return;state.materialSource=button.dataset.materialSource;
  document.querySelectorAll('[data-material-source]').forEach(b=>{b.setAttribute('aria-checked',String(b===button));b.tabIndex=b===button?0:-1;});
  $('builtin-materials-panel').hidden=state.materialSource!=='builtin';$('custom-materials-panel').hidden=state.materialSource!=='custom';invalidateResult();
}));
document.querySelector('#materials-panel .material-source-control').addEventListener('keydown',groupKeyboard);
$('materials-button').addEventListener('click',()=>$('materials-input').click());
$('materials-input').addEventListener('change',async e=>{
  const files=[...e.target.files];e.target.value='';if(state.busy || !files.length)return;
  if(files.length>8){error('最多选择 8 张素材图片。');return;}
  setBusy(true);$('generation-stage').textContent='正在读取素材';const loaded=[];
  try{for(const file of files)loaded.push(await readExtra(file,4*1024*1024));state.materialURLs.forEach(URL.revokeObjectURL);state.materialURLs=loaded.map(v=>v.url);state.materials=loaded.map(v=>v.encoded);$('materials-thumbs').replaceChildren();loaded.forEach((v,i)=>{const img=document.createElement('img');img.src=v.url;img.alt=files[i].name;img.title=files[i].name;$('materials-thumbs').append(img);});$('materials-note').textContent=`已选择 ${files.length} 张素材，重新上传可替换`;invalidateResult();}catch(err){loaded.forEach(v=>URL.revokeObjectURL(v.url));error(err.message);}finally{setBusy(false);}
});
