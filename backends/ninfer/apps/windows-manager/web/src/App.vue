<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import MonitorPanel from './MonitorPanel.vue'
import HelpTip from './HelpTip.vue'
import { groups as basicGroups, identityHelp, strictMemoryHelp, words, type Bilingual, type Language, type ParameterGroup } from './parameterHelp'
import { advancedGroups } from './advancedParameterHelp'

const groups = [...basicGroups, ...advancedGroups]

type Profile = { id:string; name:string; modelPath:string; enginePath:string; parameters:Record<string,string|null>; environment:Record<string,string|null> }
type Page = 'monitor' | 'models' | 'settings'
const clone = <T,>(value:T):T => JSON.parse(JSON.stringify(value))
const page = ref<Page>(location.pathname.startsWith('/models') ? 'models' : location.pathname.startsWith('/settings') ? 'settings' : 'monitor')
const data = ref<any>({ engine:{state:'Stopped'}, profiles:[], models:[], settings:{modelDirectories:[],language:'zh'}, recentRequests:[] })
const language = ref<Language>('zh'), languageBusy = ref(false)
const t = (zh:string,en:string) => language.value === 'zh' ? zh : en
const local = (value:Bilingual) => value[language.value]
const error = ref(''), connectionError = ref(''), notice = ref<Bilingual|null>(null), connected = ref(false), busy = ref(false), exiting = ref(false)
const displayedError = computed(() => error.value || (connectionError.value ? t('无法连接管理器，请从托盘重新打开页面。','Cannot connect to the manager. Reopen this page from the tray.')+' '+connectionError.value : ''))
const launchProfileId = ref(''), edit = ref<Profile|null>(null), dirty = ref(false), advanced = ref(false), advancedOptions = ref(false)
const jsonText = ref(''), envText = ref(''), settingsEdit = ref<any>(null), directories = ref('')
const engine = computed(() => connected.value ? (data.value.engine || {state:'Stopped'}) : {...data.value.engine,state:exiting.value ? 'Stopped' : 'Offline'})
const stateName = computed(() => ({Stopped:t('已停止','Stopped'),Starting:t('正在加载','Loading'),Running:t('运行中','Running'),Stopping:t('正在停止','Stopping'),Failed:t('启动 / 运行异常','Engine error'),Offline:t('管理器未连接','Manager offline')}[engine.value.state as string] || engine.value.state))
const active = computed(() => ['Starting','Running','Stopping'].includes(engine.value.state))
const saved = computed(() => data.value.profiles.some((p:Profile) => p.id === edit.value?.id))
const context = computed(() => Number(edit.value?.parameters['--max-context'] || 0))
const strictReserve = ref('64'), strictStep = ref('128')
const memoryMode = computed(() => {
  const value=edit.value?.parameters['--cuda-memory-policy'] ?? 'default'
  return value === 'strict' || value.startsWith('strict-') ? 'strict' : value
})
const strictPreview = computed(() => `strict-${strictReserve.value || '…'}-${strictStep.value || '…'}`)
const hybridParameters = ['--device-snapshot-slots','--cache-taps-per-request','--cache-tap-ladder','--cache-tap-min-gap','--prefix-cache-file']
const knownKeys = new Set(groups.flatMap(group => group.fields.map(field => field.key)))
const extraParameters = computed(() => Object.entries(edit.value?.parameters || {}).filter(([key]) => !knownKeys.has(key)))
const advancedParameterCount = computed(() => Object.keys(edit.value?.parameters || {}).filter(key => advancedGroups.some(group => group.fields.some(field => field.key === key))).length)
const visionEnabled = computed(() => !!edit.value && ('--vision' in edit.value.parameters || '--vision-cpu' in edit.value.parameters))
const selectedModel = computed(() => data.value.models.find((model:any) => model.path.toLowerCase().replaceAll('\\','/') === edit.value?.modelPath.toLowerCase().replaceAll('\\','/')))
function groupNotes(group:ParameterGroup):string[] {
  const parameters=edit.value?.parameters || {}, notes:string[]=[]
  const has=(key:string)=>key in parameters
  const contains=(key:string)=>group.fields.some(field=>field.key===key)
  const hybrid=memoryMode.value==='strict' || has('--use-alt-prefix-caching')
  if(group.fields.some(field => field.key === '--spec')) {
    notes.push(t('未填写的项目使用引擎或管理器默认值。切换后端会保留其他设置；下方提示帮助你处理依赖。','Blank options use engine or manager defaults. Changing the backend preserves your other settings; the notes below explain dependencies.'))
    const backend=parameters['--spec'], drafts=Number(parameters['--draft-tokens'] || 0), ngram=Number(parameters['--ngram-draft-tokens'] ?? (backend ? 15 : 0)), archive=Number(parameters['--ngram-archive-mib'] || 0), session=Number(parameters['--ngram-session-mib'] ?? 128), window=Number(parameters['--mtp-attention-window'] || 0)
    if(!backend && (drafts!==0 || ngram>0 || '--lm-head-draft' in parameters || '--adaptive-mtp' in parameters || window>0))notes.push(t('草稿后端已关闭；请清空每轮草稿数、关闭优化输出头及自适应 MTP，并把复制草稿和 MTP 窗口设为 0 或留空。上下文查找值可以保留，但当前路线只有 MTP 会使用它。','The draft backend is off. Clear drafts per round, disable the optimized head and adaptive MTP, and clear or set copied drafts and the MTP window to zero. The context lookup value may be retained, but the current path uses it only with MTP.'))
    if(backend && (drafts<1 || drafts>15))notes.push(t('启用草稿后端后，每轮草稿数必须为 1–15，不能留空。','With a backend enabled, drafts per round must be set to 1–15.'))
    if(backend && backend!=='mtp' && ('--adaptive-mtp' in parameters || window>0))notes.push(t('自适应 MTP 与非零 MTP 注意力窗口仅适用于 mtp。','Adaptive MTP and a nonzero MTP attention window require mtp.'))
    if(backend==='mtp' && window>0 && window<drafts+1)notes.push(t('MTP 注意力窗口至少要比每轮草稿数多 1。','The MTP attention window must be at least drafts per round + 1.'))
    if(ngram>15 && Number(parameters['--max-concurrency'] ?? 1)!==1)notes.push(t('复制草稿超过 15 时，并行请求数必须为 1。','More than 15 copied drafts requires concurrency 1.'))
    if(archive>0 && (ngram<=0 || session<1 || session>archive))notes.push(t('跨请求复制来源需要启用复制草稿；每会话预算须在 1 MiB 到总预算之间。','The copy archive requires ngram drafting and a per-session budget from 1 MiB to the total archive budget.'))
    if('--ngram-native-sessions' in parameters && archive<=0)notes.push(t('识别客户端会话标识需要大于 0 的跨请求复制来源预算。','Client session recognition requires a positive copy archive budget.'))
    if(Number(parameters['--lookup-ngram'] || 0)>0 && backend!=='mtp')notes.push(t('上下文查找值已保留；当前未选择 MTP，因此不会执行这项查找草稿加速。','The context lookup value is retained. It is inactive because MTP is not selected.'))
  }
  if(group.fields.some(field => field.key === '--vision')) {
    notes.push(t('视觉默认关闭。当前 Swift XXS / S 的 text + MTP 文件不包含 Vision；只有换用包含视觉组件的模型后才能接收图片或视频。','Vision is off by default. The current Swift XXS / S text + MTP artifacts do not include Vision; image or video input needs a model with a Vision component.'))
    if(visionEnabled.value && selectedModel.value && !(selectedModel.value.components || []).some((component:string) => component.toLowerCase()==='vision'))notes.push(t('所选模型的扫描结果没有 Vision 组件。请更换包含视觉组件的模型，或关闭视觉。','The selected model’s scan found no Vision component. Select a Vision-capable model or disable vision.'))
    if(visionEnabled.value && memoryMode.value!=='default')notes.push(t('当前显存策略仅支持纯文本。启用视觉前，请手动把显存策略改为 default；此处不会替你改动。','The current memory policy is text-only. Select default memory policy yourself before enabling vision.'))
    if(!visionEnabled.value && ['overlay','cpu'].includes(parameters['--vision-residency'] || ''))notes.push(t('已保留视觉放置位置；overlay / cpu 需要开启视觉。若保持视觉关闭，请把放置位置改为 resident 或引擎默认。','The saved placement is preserved. overlay / cpu require vision. To leave vision off, select resident or the engine default.'))
  }
  if(contains('--use-alt-prefix-caching')) {
    if(memoryMode.value==='strict')notes.push(t('当前 strict 已自动选择混合前缀缓存。传统缓存与关闭前缀复用不适用于此模式；更换路线需要先调整显存策略。','Strict currently selects hybrid prefix caching automatically. Traditional caching and disabled prefix reuse do not apply; change the memory policy before choosing another route.'))
    if(has('--kv-headroom-mib') && memoryMode.value!=='default')notes.push(t('已填写自动 KV 规划预留，但此项仅适用于 default 显存策略。strict 的余量在常用选项的显存策略中设置。','Auto-KV headroom is set but requires default memory policy. Set the strict reserve in the common memory-policy controls instead.'))
  }
  if(contains('--device-state-slots') || contains('--disk-kv-path')) {
    notes.push(hybrid?t('这些项目属于传统缓存。当前使用混合缓存，填写之前请先切换缓存路线；收起此分组不会删除已保存值。','These options belong to the traditional cache. Hybrid caching is currently selected; change the cache route before setting them. Collapsing the group preserves saved values.'):t('这些项目用于传统前缀缓存；统一 Host 缓存预算与独立 Host 槽位 / KV 预算只能选择一种。','These options apply to the traditional prefix cache. Choose either the unified Host budget or the separate Host slots / KV budgets.'))
  }
  if(contains('--cache-taps-per-request') && !hybrid)notes.push(t('这些项目需要混合前缀缓存。填写时会自动启用该缓存路线；它与显存不足时借用系统内存无关。','These options need hybrid prefix caching. Setting one enables that cache route automatically; this is separate from borrowing system RAM when VRAM is insufficient.'))
  if(contains('--no-cuda-graph') && has('--no-cuda-graph') && Number(parameters['--cuda-graph-allowance-mib'] || 0)>0)notes.push(t('已关闭 CUDA Graph，请把常用选项中的 Graph 启动预算设为 0 或清空。','CUDA Graphs are disabled. Clear the common Graph startup budget or set it to zero.'))
  if(contains('--no-thinking') && has('--no-thinking') && parameters['--default-reasoning-effort'] && parameters['--default-reasoning-effort']!=='none')notes.push(t('默认关闭思考与当前思考强度冲突；请在常用选项中清空思考强度或选 none。','Disabling thinking conflicts with the current default effort. Clear it in the common controls or select none.'))
  if(contains('--assistant-prefill') && has('--assistant-prefill') && !has('--no-thinking'))notes.push(t('续写末尾 assistant 消息需要默认关闭思考；可在“思考预算与回答阶段采样”中设置。','Continuing a trailing assistant message requires thinking disabled by default; set it in Thinking budgets and answer-stage sampling.'))
  return notes
}
function validateNumericFields(parameters:Record<string,string|null>) {
  for(const group of groups)for(const field of group.fields) {
    if(field.inputType!=='number' || !(field.key in parameters))continue
    const raw=parameters[field.key], value=Number(raw)
    if(raw===null || raw.trim()==='' || !Number.isFinite(value) ||
       (field.min!==undefined && value<field.min) || (field.max!==undefined && value>field.max) ||
       (field.step!==undefined && Number.isInteger(field.step) && (!Number.isInteger(value) || (value-(field.min ?? 0))%field.step!==0))) {
      const range=[field.min,field.max].filter(bound=>bound!==undefined).join('–')
      const whole=field.step!==undefined && Number.isInteger(field.step)
      throw new Error(t(`${local(field.label)}（${field.key}）请输入${range}${whole?' 范围内的整数':' 范围内的数值'}${field.step && field.step>1?`，步长 ${field.step}`:''}，或留空使用默认值。`,`${local(field.label)} (${field.key}): enter ${whole?'an integer':'a number'} in ${range}${field.step && field.step>1?`, step ${field.step}`:''}, or leave blank for the default.`))
    }
  }
}
const n = (value:unknown,digits=1) => value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString(language.value === 'zh' ? 'zh-CN' : 'en-US',{maximumFractionDigits:digits})
let timer:ReturnType<typeof setInterval> | undefined
let languageVersion = 0
watch(language,value => { document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en'; document.title = value === 'zh' ? 'NInfer · 本地模型管理器' : 'NInfer · Local Model Manager' },{immediate:true})
function runningParam(key:string) { const args=engine.value.launch?.arguments || []; const i=args.indexOf(key); return i >= 0 ? args[i+1] : data.value.profiles.find((p:Profile) => p.id === engine.value.profileId)?.parameters[key] }
function nav(next:Page) { page.value=next; window.history.replaceState({},'',next === 'monitor' ? '/' : '/'+next); if(next === 'settings' && !settingsEdit.value)resetSettings() }
async function api(path:string,method='GET',body?:unknown) {
  const response = await fetch('/api'+path,{method,headers:{'Content-Type':'application/json','X-NInfer-Manager':'1'},body:body === undefined ? undefined : JSON.stringify(body)})
  const text=await response.text(); let payload:any; try{payload=JSON.parse(text)}catch{payload=text}
  if(!response.ok)throw new Error(typeof payload === 'string' ? payload : payload.error || payload.message || `HTTP ${response.status}`)
  return payload
}
async function poll() {
  const version=languageVersion
  try {
    const next=await api('/state'); data.value=next; connected.value=true; connectionError.value=''
    if(!languageBusy.value && version === languageVersion && ['zh','en'].includes(next.settings?.language))language.value=next.settings.language
    if(page.value === 'settings' && !settingsEdit.value)resetSettings()
    if(!launchProfileId.value && next.profiles?.length)launchProfileId.value=next.settings.defaultProfileId || next.profiles[0].id
    if(!edit.value && next.profiles?.length)choose(next.settings.defaultProfileId || next.profiles[0].id,false)
  } catch(ex) {
    connected.value=false
    if(exiting.value){error.value='';connectionError.value='';notice.value=words('管理器连接已关闭。重新双击 NInferManager.exe 可启动。','The manager has closed. Open NInferManager.exe to start it again.');clearInterval(timer)}
    else connectionError.value=String((ex as Error).message)
  }
}
async function changeLanguage(next:Language) {
  if(languageBusy.value || next === language.value)return
  const previous=language.value; languageBusy.value=true; languageVersion++; language.value=next; error.value=''
  try {
    await api('/language','PUT',{language:next})
    data.value.settings.language=next
    if(settingsEdit.value)settingsEdit.value.language=next
  } catch(ex) { language.value=previous; error.value=t('语言设置未保存：','Language was not saved: ')+(ex as Error).message }
  finally { languageVersion++; languageBusy.value=false }
}
async function action(fn:()=>Promise<unknown>,message?:Bilingual) {
  busy.value=true; error.value=''
  try { await fn(); if(message)notice.value=message; await poll() } catch(ex) { error.value=(ex as Error).message } finally { busy.value=false }
}
async function exitManager() {
  if(!window.confirm(t('停止模型并退出管理器？','Stop the model and exit the manager?')))return
  exiting.value=true; error.value=''; notice.value=words('正在停止模型并退出管理器…','Stopping the model and exiting…')
  try { await api('/exit','POST') } catch(ex) { error.value=(ex as Error).message }
  await poll()
}
function choose(id:string,check=true) {
  if(check && dirty.value && !window.confirm(t('当前有未保存修改，放弃这些修改？','Discard the unsaved changes to this profile?')))return
  const profile=data.value.profiles.find((p:Profile) => p.id === id); if(!profile)return
  edit.value=clone(profile); dirty.value=false; syncJson()
}
function syncJson() { jsonText.value=JSON.stringify(edit.value?.parameters || {},null,2); envText.value=JSON.stringify(edit.value?.environment || {},null,2);syncMemoryEditor() }
function syncMemoryEditor() {
  const match=/^strict-([0-9]+)-([0-9]+)$/.exec(edit.value?.parameters['--cuda-memory-policy'] || '')
  strictReserve.value=match?.[1] || '64';strictStep.value=match?.[2] || '128'
}
function strictValue() { return strictReserve.value === '64' && strictStep.value === '128' ? 'strict' : `strict-${strictReserve.value}-${strictStep.value}` }
function setMemoryMode(value:string) {
  if(!edit.value || !['default','mixed','strict'].includes(value))return
  if(value === 'strict')delete edit.value.parameters['--use-alt-prefix-caching']
  else if(hybridParameters.some(key => key in edit.value!.parameters))edit.value.parameters['--use-alt-prefix-caching']=null
  setParam('--cuda-memory-policy',value === 'strict' ? strictValue() : value)
}
function setStrictValue(part:'reserve'|'step',value:string) {
  if(part === 'reserve')strictReserve.value=value;else strictStep.value=value
  setParam('--cuda-memory-policy',strictValue())
}
function validateMemoryPolicy(parameters:Record<string,string|null>) {
  if(!('--cuda-memory-policy' in parameters))return
  const value=parameters['--cuda-memory-policy']
  if(value === 'default' || value === 'mixed' || value === 'strict')return
  const match=/^strict-([0-9]+)-([0-9]+)$/.exec(value || '')
  if(!match || match[0] !== value || !Number.isSafeInteger(Number(match[1])) || Number(match[1]) > 17592186044415 || !Number.isInteger(Number(match[2])) || Number(match[2]) < 1 || Number(match[2]) > 16384)throw new Error(t('显存策略须为 default、mixed、strict 或 strict-余量-步长。余量 0–17592186044415 MiB，步长 1–16384 MiB。','Memory policy must be default, mixed, strict or strict-reserve-step. Reserve: 0–17592186044415 MiB; step: 1–16384 MiB.'))
  parameters['--cuda-memory-policy']=Number(match[1]) === 64 && Number(match[2]) === 128 ? 'strict' : `strict-${Number(match[1])}-${Number(match[2])}`
}
function setParam(key:string,value:string) {
  if(!edit.value)return
  if(value === '')delete edit.value.parameters[key]
  else {
    edit.value.parameters[key]=value
    if(hybridParameters.includes(key) && memoryMode.value !== 'strict')edit.value.parameters['--use-alt-prefix-caching']=null
  }
  if(key==='--vision-residency' && value!=='cpu')delete edit.value.parameters['--vision-cpu']
  dirty.value=true
}
function setFlag(key:string,value:boolean) {
  if(!edit.value)return
  if(value)edit.value.parameters[key]=null; else delete edit.value.parameters[key]
  if(key==='--vision-cpu') {
    if(value)edit.value.parameters['--vision']=null
    edit.value.parameters['--vision-residency']=value ? 'cpu' : 'resident'
  }
  if(key==='--vision' && !value)delete edit.value.parameters['--vision-cpu']
  dirty.value=true
}
function newProfile() {
  if(dirty.value && !window.confirm(t('放弃当前未保存修改？','Discard the current unsaved changes?')))return
  edit.value=data.value.profiles.length ? clone(data.value.profiles[0]) : {id:'',name:'',modelPath:data.value.models[0]?.path || '',enginePath:'engine/ninfer-serve.exe',parameters:{},environment:{}}
  edit.value!.id=crypto.randomUUID().replaceAll('-',''); edit.value!.name=t('新的模型配置','New model profile'); dirty.value=true; syncJson()
}
function objectText(text:string):Record<string,string|null> {
  let result:unknown
  try{result=JSON.parse(text)}catch{throw new Error(t('JSON 格式错误，请检查引号、逗号和括号。','Invalid JSON. Check quotes, commas and braces.'))}
  if(result === null || Array.isArray(result) || typeof result !== 'object' || Object.values(result).some(value => value !== null && typeof value !== 'string'))throw new Error(t('JSON 必须是对象，每个值必须是字符串或 null。','JSON must be an object with string or null values.'))
  return result as Record<string,string|null>
}
function applyAdvanced() { if(edit.value){const parameters=objectText(jsonText.value),environment=objectText(envText.value);validateMemoryPolicy(parameters);edit.value.parameters=parameters;edit.value.environment=environment;syncMemoryEditor()} }
function toggleAdvanced() { error.value=''; if(advanced.value){try{applyAdvanced()}catch(ex){error.value=(ex as Error).message;return}}else syncJson(); advanced.value=!advanced.value }
function duplicate() {
  if(!edit.value)return
  try{if(advanced.value)applyAdvanced()}catch(ex){error.value=(ex as Error).message;return}
  edit.value=clone(edit.value); edit.value.id=crypto.randomUUID().replaceAll('-',''); edit.value.name+=t(' · 副本',' · Copy'); dirty.value=true; syncJson()
}
async function save() {
  if(!edit.value)return
  await action(async() => { if(advanced.value)applyAdvanced(); validateMemoryPolicy(edit.value!.parameters);validateNumericFields(edit.value!.parameters);await api('/profiles/'+edit.value!.id,'PUT',edit.value);dirty.value=false;syncJson() },words('配置已保存，下次启动生效。','Profile saved. Changes apply on the next start.'))
}
async function remove() {
  if(!edit.value || !saved.value || !window.confirm(t(`删除配置“${edit.value.name}”？模型文件会保留。`,`Delete “${edit.value.name}”? The model files will be kept.`)))return
  await action(async() => { const id=edit.value!.id;await api('/profiles/'+id,'DELETE');edit.value=null;dirty.value=false;if(launchProfileId.value === id)launchProfileId.value='' },words('配置已删除。','Profile deleted.'))
}
function resetSettings() { settingsEdit.value=clone(data.value.settings);settingsEdit.value.language=language.value;directories.value=(settingsEdit.value.modelDirectories || []).join('\n') }
async function saveSettings() {
  await action(async() => { const value={...settingsEdit.value,language:language.value,modelDirectories:directories.value.split('\n').map(s=>s.trim()).filter(Boolean)};await api('/settings','PUT',value);settingsEdit.value=clone(value) },words('设置已保存。','Settings saved.'))
}
async function makeDefault() {
  if(!edit.value || dirty.value)return
  await action(async() => {await api('/settings','PUT',{...data.value.settings,language:language.value,defaultProfileId:edit.value!.id})},words('已设为默认启动配置。','Default startup profile updated.'))
}
async function copy(value:string) { try{await navigator.clipboard.writeText(value);notice.value=words('已复制。','Copied.')}catch{notice.value=words(value,value)} }
onMounted(async() => {await poll();if(!exiting.value)timer=setInterval(poll,1800)})
onUnmounted(() => clearInterval(timer))
</script>

<template>
 <div class="shell">
  <aside>
   <div class="brand"><span class="brand-mark" aria-hidden="true">N</span><div>NInfer<small>{{t('本地模型管理器','LOCAL MODEL MANAGER')}}</small></div></div>
   <nav :aria-label="t('主要导航','Main navigation')">
    <button :class="{chosen:page==='monitor'}" @click="nav('monitor')"><span aria-hidden="true">◉</span>{{t('运行监控','Monitor')}}</button>
    <button :class="{chosen:page==='models'}" @click="nav('models')"><span aria-hidden="true">▧</span>{{t('模型与配置','Models & profiles')}}</button>
    <button :class="{chosen:page==='settings'}" @click="nav('settings')"><span aria-hidden="true">⚙</span>{{t('偏好设置','Preferences')}}</button>
   </nav>
   <div class="sidebar-bottom"><span class="dot" :class="{online:connected}"></span>{{connected?t('管理器已连接','Manager connected'):t('管理器未连接','Manager offline')}}<p>{{t('单模型 · 本机运行','One model · Local inference')}}<br>{{t('关闭网页不影响推理','Closing this page keeps inference running')}}</p></div>
  </aside>
  <main>
   <header><div><div class="eyebrow">{{t('你的本地推理工作台','YOUR LOCAL INFERENCE WORKSPACE')}}</div><h1>{{page==='monitor'?t('运行监控','Monitor'):page==='models'?t('模型与启动配置','Models & launch profiles'):t('偏好设置','Preferences')}}</h1></div><div class="header-actions"><div class="language-switch" role="group" :aria-label="t('界面语言','Interface language')"><button :class="{selected:language==='zh'}" :aria-pressed="language==='zh'" :disabled="languageBusy||!connected" @click="changeLanguage('zh')">中文</button><button :class="{selected:language==='en'}" :aria-pressed="language==='en'" :disabled="languageBusy||!connected" @click="changeLanguage('en')">English</button></div><span class="status" :class="engine.state.toLowerCase()"><span class="dot"></span>{{stateName}}</span></div></header>
   <div v-if="displayedError" class="alert error" role="alert"><span>{{displayedError}}</span><button :aria-label="t('关闭提示','Dismiss message')" @click="error='';connectionError=''">×</button></div>
   <div v-if="notice" class="alert success" role="status"><span>{{local(notice)}}</span><button :aria-label="t('关闭提示','Dismiss message')" @click="notice=null">×</button></div>
   <section class="running-card">
    <div><div class="eyebrow">{{t('当前引擎','CURRENT ENGINE')}}</div><h2>{{engine.profileName||t('等待启动模型','Ready to start a model')}}</h2><p v-if="engine.state==='Running'">PID {{engine.pid}} · {{n(Number(runningParam('--max-context'))/1024)}}K {{t('上下文','context')}} · {{runningParam('--kv-dtype')}}</p><p v-else-if="engine.state==='Starting'">{{t('正在载入权重并检查显存驻留，请稍候。','Loading weights and checking GPU residency. Please wait.')}}</p><p v-else>{{t('选择已保存的配置，即可启动本地 API。','Choose a saved profile to start the local API.')}}</p></div>
    <div class="actions"><select v-if="!active" v-model="launchProfileId" :aria-label="t('选择启动配置','Choose a launch profile')"><option v-for="p in data.profiles" :key="p.id" :value="p.id">{{p.name}}</option><option v-if="!data.profiles.length" value="">{{t('暂无配置','No profiles')}}</option></select><button v-if="!active" class="primary" :disabled="busy||!connected||!launchProfileId" @click="action(()=>api('/start/'+launchProfileId,'POST'))">{{t('启动模型','Start model')}}</button><button v-else class="danger-outline" :disabled="busy||!connected||engine.state==='Stopping'" @click="action(()=>api('/stop','POST'))">{{engine.state==='Stopping'?t('正在停止…','Stopping…'):t('停止服务','Stop service')}}</button></div>
    <div v-if="engine.state==='Running'" class="api-row"><button @click="copy(engine.apiBase)"><small>API BASE</small>{{engine.apiBase}} <span>{{t('复制','Copy')}} ↗</span></button><button @click="copy(engine.modelId)"><small>MODEL ID</small>{{engine.modelId}} <span>{{t('复制','Copy')}} ↗</span></button></div><div v-if="engine.error" class="inline-error">{{engine.error}}</div>
   </section>
   <MonitorPanel v-if="page==='monitor'" :data="{...data, engine}" :language="language" />

   <template v-if="page==='models'">
    <section class="panel"><div class="panel-title"><div><h3>{{t('模型目录','Available models')}}</h3><p class="hint">{{t('扫描 .ninfer 主文件并读取元数据；不加载 GPU。','Scans .ninfer entry files and reads metadata without loading the GPU.')}}</p></div><button :disabled="busy||!connected" @click="action(()=>api('/models/scan','POST'),words('模型目录已刷新。','Model list refreshed.'))">{{t('重新扫描','Rescan')}}</button></div><div class="model-grid"><article class="model-card" v-for="model in data.models" :key="model.path"><div class="badge" :class="{ok:model.isComplete}">{{model.isComplete?t('文件就绪','Files ready'):t('文件异常','File issue')}}</div><h4>{{model.name}}</h4><p>{{n(model.fileBytes/1073741824,2)}} GiB · {{(model.components||[]).join(' + ')}} · {{t('声明','Declared')}} {{n(model.maxContext/1024)}}K</p><code>{{model.path}}</code><p v-if="model.error" class="inline-error">{{model.error}}</p></article><p v-if="!data.models?.length" class="empty">{{connected?t('未发现模型。请在偏好设置中添加模型所在目录。','No models found. Add their directories in Preferences.'):t('连接管理器后显示模型。','Models appear when the manager connects.')}}</p></div></section>
    <section class="panel"><div class="panel-title"><h3>{{t('启动配置','Launch profiles')}} <span class="count">{{data.profiles.length}}</span></h3><button :disabled="!connected" @click="newProfile">＋ {{t('新建配置','New profile')}}</button></div><div class="profile-tabs"><button v-for="p in data.profiles" :key="p.id" :class="{selected:edit?.id===p.id}" @click="choose(p.id)">{{p.name}}<span v-if="p.id===data.settings.defaultProfileId">{{t('默认','Default')}}</span></button></div>
     <div v-if="edit">
      <div class="editor-head"><span>{{dirty?t('有未保存的修改','Unsaved changes'):t('配置已保存','Profile saved')}} · {{t('参数在下次启动时生效','Changes apply on the next start')}}</span><div class="actions"><button @click="duplicate">{{t('复制配置','Duplicate')}}</button><button :disabled="busy||!connected||dirty||!saved" @click="makeDefault">{{t('设为默认','Make default')}}</button><button class="danger-text" :disabled="busy||!connected||!saved||edit.id===engine.profileId" @click="remove">{{t('删除配置','Delete')}}</button></div></div>
      <div class="form-grid profile-identity">
       <div class="field"><div class="field-label"><label for="profile-name">{{t('配置名称','Profile name')}}</label><HelpTip :title="t('配置名称','Profile name')" :text="local(identityHelp.name)" :language="language" /></div><input id="profile-name" v-model="edit.name" @input="dirty=true"></div>
       <div class="field"><div class="field-label"><label for="model-path">{{t('模型文件','Model file')}}</label><HelpTip :title="t('模型文件','Model file')" :text="local(identityHelp.model)" parameter="MODEL" :language="language" restart /></div><select id="model-path" v-model="edit.modelPath" @change="dirty=true"><option :value="edit.modelPath">{{edit.modelPath||t('选择模型文件','Choose a model file')}}</option><option v-for="m in data.models.filter((m:any)=>m.path!==edit!.modelPath)" :key="m.path" :value="m.path">{{m.name}}</option></select></div>
       <div class="field"><div class="field-label"><label for="engine-path">{{t('引擎路径','Engine path')}}</label><HelpTip :title="t('引擎路径','Engine path')" :text="local(identityHelp.engine)" :language="language" restart /></div><input id="engine-path" v-model="edit.enginePath" @input="dirty=true"></div>
      </div>
      <div class="profile-parameters">
       <p v-if="advanced" class="hint">{{t('专家 JSON 编辑已打开，下面的参数表暂时只读。收起 JSON 后会将修改应用到表单，点击保存才会写入配置。','Expert JSON editing is open, so the parameter form is read-only. Closing JSON applies your edits to the form; Save writes the profile.')}}</p>
       <template v-for="(group,index) in groups" :key="index">
       <button v-if="index===basicGroups.length" class="advanced-options-toggle" :aria-expanded="advancedOptions" :aria-controls="advancedGroups.map((_,i)=>'advanced-parameter-group-'+i).join(' ')" @click="advancedOptions=!advancedOptions"><span>{{advancedOptions?'▾':'▸'}} {{t('高级选项','Advanced options')}}</span><small>{{t('服务、缓存、内核、请求限制与日志','Serving, caches, kernels, request limits and logs')}} · {{advancedParameterCount}} {{t('项已配置','configured')}}</small></button>
       <section v-show="!group.advanced||advancedOptions" :id="group.advanced?'advanced-parameter-group-'+(index-basicGroups.length):undefined" class="parameter-group" :class="{'advanced-parameter-group':group.advanced}"><h4>{{local(group.title)}}</h4><div v-if="groupNotes(group).length" class="parameter-notes"><p v-for="note in groupNotes(group)" :key="note">{{note}}</p></div><div class="form-grid">
        <div v-for="field in group.fields" :key="field.key" class="field" :class="{'flag-field':field.flag,'memory-policy-field':field.key==='--cuda-memory-policy'}">
         <div class="field-label"><label :for="field.key">{{local(field.label)}}<span v-if="field.key==='--max-context'" class="muted"> · {{n(context/1024)}}K</span></label><HelpTip :title="local(field.label)" :text="local(field.help)" :parameter="field.key" :language="language" restart /></div><code class="field-option">{{field.key}}</code>
         <template v-if="field.key==='--cuda-memory-policy'">
          <select :id="field.key" :value="memoryMode" :disabled="advanced" @change="setMemoryMode(($event.target as HTMLSelectElement).value)"><option value="default">{{t('默认策略','Default policy')}}</option><option value="mixed">{{t('允许借用系统内存','Allow borrowing system RAM')}}</option><option value="strict">{{t('仅使用独立显存','Dedicated VRAM only')}}</option></select>
          <div v-if="memoryMode==='strict'" class="strict-settings">
           <div class="field"><div class="field-label"><label for="strict-reserve">{{t('显存余量 · MiB','VRAM reserve · MiB')}}</label><HelpTip :title="t('显存余量','VRAM reserve')" :text="local(strictMemoryHelp.reserve)" parameter="--cuda-memory-policy strict-RESERVE-STEP" :language="language" restart /></div><input id="strict-reserve" type="number" min="0" max="17592186044415" step="1" :disabled="advanced" :value="strictReserve" @input="setStrictValue('reserve',($event.target as HTMLInputElement).value)"></div>
           <div class="field"><div class="field-label"><label for="strict-step">{{t('探测步长 · MiB','Probe step · MiB')}}</label><HelpTip :title="t('探测步长','Probe step')" :text="local(strictMemoryHelp.step)" parameter="--cuda-memory-policy strict-RESERVE-STEP" :language="language" restart /></div><input id="strict-step" type="number" min="1" max="16384" step="1" :disabled="advanced" :value="strictStep" @input="setStrictValue('step',($event.target as HTMLInputElement).value)"></div>
           <p class="policy-preview">{{t('对应参数','Policy value')}} <code>{{strictPreview}}</code><span v-if="strictValue()==='strict'">{{t('（64/128 保存为 strict）','(64/128 is saved as strict)')}}</span></p>
          </div>
          <p v-if="memoryMode==='strict'" class="hint">{{t('自动选择所需上下文缓存实现。CPU Host cache 独立配置，不属于显存不足时的系统内存借用。','Automatically selects the required context-cache implementation. CPU Host cache is independent and is not system-RAM borrowing caused by a VRAM shortfall.')}}</p>
          <p v-else-if="memoryMode==='mixed'" class="hint">{{t('显存不足时允许 Windows 用系统内存承接 CUDA 设备数据，放置由驱动决定，并非强制分配到 Shared。可能降低速度；auto 最多为每个并发规划一个完整上下文窗口，仍可能分配失败。','When VRAM is insufficient, Windows may use system RAM for CUDA device data. The driver decides placement; Shared allocation is not forced. Performance may decrease. Auto plans at most one complete context window per concurrent request, and allocation may still fail.')}}</p>
          <p v-else class="hint">{{t('保留按 CUDA free 规划的原分配行为，不做严格驻留探测；也不保证绝不使用共享系统内存。','Keeps the original CUDA-free planning behavior without strict residency probing. It does not guarantee that shared system memory will never be used.')}}</p>
          <p v-if="memoryMode!=='strict'&&'--use-alt-prefix-caching' in edit.parameters" class="hint">{{t('已保留混合前缀缓存以继续使用现有快照参数；高级选项中可查看。这是缓存路线，不是 Shared 显存许可。','Hybrid prefix caching is retained for the existing snapshot options; it is visible in Advanced options. This cache route is separate from permission to use Shared GPU memory.')}}</p>
         </template>
         <label v-else-if="field.flag" class="flag-value"><input :id="field.key" type="checkbox" :disabled="advanced||field.readOnly" :checked="field.key==='--vision'?visionEnabled:field.key in edit.parameters" @change="setFlag(field.key,($event.target as HTMLInputElement).checked)"><span>{{(field.key==='--vision'?visionEnabled:field.key in edit.parameters)?t('已开启','Enabled'):t('已关闭','Disabled')}}</span></label>
         <select v-else-if="field.choices" :id="field.key" :value="edit.parameters[field.key]??''" :disabled="advanced||field.readOnly" @change="setParam(field.key,($event.target as HTMLSelectElement).value)"><option v-if="edit.parameters[field.key]&&!field.choices.some(option=>option.value===edit!.parameters[field.key])" :value="edit.parameters[field.key]!">{{t('已保存值，请核对：','Saved value, please check: ')}}{{edit.parameters[field.key]}}</option><option v-for="option in field.choices" :key="option.value" :value="option.value">{{local(option.label)}}</option></select>
         <input v-else :id="field.key" :type="field.inputType||'text'" :min="field.min" :max="field.max" :step="field.step" :disabled="advanced||field.readOnly" :value="edit.parameters[field.key]??''" :placeholder="field.placeholder||t('留空使用默认值','Leave blank for default')" :autocomplete="field.inputType==='password'?'new-password':undefined" @input="setParam(field.key,($event.target as HTMLInputElement).value)">
         <small v-if="field.defaultValue" class="field-default">{{t('未填写时：','When omitted: ')}}{{local(field.defaultValue)}}</small>
         <small v-if="field.readOnly" class="field-default">{{t('当前 Windows 管理器不支持启用此参数。','This option cannot be enabled in the Windows manager.')}}</small>
        </div>
       </div></section>
       </template>
       <section v-if="extraParameters.length" class="parameter-group"><h4>{{t('其他已保存参数','Other saved parameters')}}</h4><p class="hint">{{t('这些参数会原样保留。具体作用请查所选引擎 --help，可在专家 JSON 中编辑。','These options are preserved as written. Consult the selected engine’s --help for their meaning and edit them in expert JSON.')}}</p><div class="extra-parameters"><code v-for="[key,value] in extraParameters" :key="key">{{key}}{{value===null?'':' '+value}}</code></div></section>
      </div>
      <button class="link" :aria-expanded="advanced" @click="toggleAdvanced">{{advanced?t('收起专家 JSON 并应用到表单','Close expert JSON and apply to form'):t('专家编辑：完整参数与环境变量 JSON','Expert editor: complete parameters and environment JSON')}}</button>
      <div v-if="advanced" class="two-col advanced"><div class="field"><div class="field-label"><label for="parameters-json">{{t('完整启动参数 JSON','Complete launch parameters · JSON')}}</label><HelpTip :title="t('完整启动参数 JSON','Launch parameter JSON')" :text="local(identityHelp.json)" :language="language" restart /></div><textarea id="parameters-json" v-model="jsonText" @input="dirty=true" spellcheck="false"></textarea></div><div class="field"><div class="field-label"><label for="environment-json">{{t('环境变量 JSON','Environment · JSON')}}</label><HelpTip :title="t('环境变量','Environment variables')" :text="local(identityHelp.environment)" :language="language" restart /></div><textarea id="environment-json" v-model="envText" @input="dirty=true" spellcheck="false"></textarea></div></div>
      <div class="save-bar"><p>{{t('每次保存保留修订记录。修改配置不会立即重启服务。','Each save keeps a revision. Editing a profile does not restart the running service.')}}</p><button class="primary" :disabled="busy||!connected||!dirty" @click="save">{{busy?t('正在保存…','Saving…'):t('保存配置','Save profile')}}</button></div>
     </div>
     <p v-else class="empty">{{t('新建配置以选择模型和启动参数。','Create a profile to choose a model and launch options.')}}</p>
    </section>
   </template>

   <template v-if="page==='settings'">
    <section class="panel settings" v-if="settingsEdit"><h3>{{t('自动启动','Automatic startup')}}</h3>
     <label class="setting-row"><div><strong>{{t('随 Windows 登录启动','Start at Windows sign-in')}}</strong><p>{{t('登录后只显示托盘，不弹出控制台或浏览器。','Starts in the tray without opening a console or browser.')}}</p></div><input type="checkbox" v-model="settingsEdit.startWithWindows"></label>
     <label class="setting-row"><div><strong>{{t('自动加载默认模型','Automatically load the default model')}}</strong><p>{{t('管理器启动后载入下方指定配置。','Loads the selected profile when the manager starts.')}}</p></div><input type="checkbox" v-model="settingsEdit.autoStartModel"></label>
     <label>{{t('默认启动配置','Default startup profile')}}<select v-model="settingsEdit.defaultProfileId"><option v-for="p in data.profiles" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
     <h3>{{t('模型扫描目录','Model scan directories')}}</h3><p class="hint">{{t('每行一个目录；相对路径以模型包根目录为起点。','One directory per line. Relative paths start at the package root.')}}</p><label for="model-directories" class="sr-only">{{t('模型扫描目录','Model scan directories')}}</label><textarea id="model-directories" v-model="directories" rows="5"></textarea>
     <h3>{{t('本地管理端口','Management web port')}}</h3><div class="field"><div class="field-label"><label for="web-port">{{t('端口 · 重启管理器后生效','Port · applies after restarting the manager')}}</label><HelpTip :title="t('本地管理端口','Management web port')" :text="t('管理页面使用的端口，范围 1024–65535，必须与模型 API 端口不同。只改变管理网页地址；保存后重启管理器才生效，当前网页不会立即迁移。','The management web port, from 1024 to 65535. It must differ from the model API port. This changes only the management address. Restart the manager after saving; this page does not move immediately.')" :language="language" /></div><input id="web-port" type="number" v-model.number="settingsEdit.webPort" min="1024" max="65535"></div>
     <div class="save-bar"><button @click="resetSettings">{{t('还原未保存修改','Discard unsaved changes')}}</button><button class="primary" :disabled="busy||!connected||languageBusy" @click="saveSettings">{{busy?t('正在保存…','Saving…'):t('保存设置','Save settings')}}</button></div>
    </section>
    <p v-else class="empty">{{t('正在读取设置…','Loading settings…')}}</p>
    <section class="panel settings"><h3>{{t('退出管理器','Exit manager')}}</h3><p class="hint">{{t('退出会同时停止模型和本地监控；仅关闭浏览器则继续运行。','Exiting stops the model and local monitor. Closing only the browser keeps them running.')}}</p><button class="danger-outline" :disabled="!connected||exiting" @click="exitManager">{{exiting?t('正在退出…','Exiting…'):t('停止并退出','Stop and exit')}}</button></section>
   </template>
   <footer>NInfer · CUDA Native <span>{{t('所有配置与模型保存在本机','All profiles and models stay on this computer')}}</span></footer>
  </main>
 </div>
</template>
