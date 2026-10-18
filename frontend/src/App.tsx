import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft, ArrowRight, BarChart3, BookOpen, Check, ChevronRight, CircleHelp, CloudUpload, Cpu, Download, ExternalLink, FileImage, FlaskConical, Grid2X2, ImagePlus, Leaf, LoaderCircle, Maximize2, Microscope, ScanLine, Search, ShieldCheck, Sprout, Trash2, X } from 'lucide-react';
import type { Health, ModelId, ModelInfo, Prediction } from './types';
import { API_BASE, getStatus, predict } from './api';
import { bytes, diseases, diseaseSource, percent } from './data';

type Page = 'predict' | 'library' | 'benchmark' | 'docs';
const nav: [Page, string][] = [['predict', 'Predict / Analyze'], ['library', 'Disease Library'], ['benchmark', 'Model Benchmark'], ['docs', 'System Docs']];
const fallbackModels: ModelInfo[] = [
  { id: 'simple_cnn', name: 'Simple CNN', description: 'Trained from scratch · 3 convolution blocks' },
  { id: 'mobilenetv3', name: 'MobileNetV3Large', description: 'ImageNet transfer learning · fine-tuned backbone' },
].map(m => ({ ...m, id: m.id as ModelId, available: false, status_message: null, class_names: [], input_shape: [1,224,224,3], preprocessing: {}, parameter_count: null, size_bytes: null, sha256: null, metrics: null }));

function Notice() {
  return <div className="notice"><ShieldCheck size={22}/><div><h3>Academic research · interpret with care</h3><p>These models return uncalibrated scores across six known classes. Non-rice images, mixed infections, and unsupported conditions may still receive high scores. Confirm findings with a local agricultural specialist before making treatment decisions.</p></div></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>(() => nav.some(([p]) => `#${p}` === location.hash) ? location.hash.slice(1) as Page : 'predict');
  const [models, setModels] = useState<ModelInfo[]>(fallbackModels);
  const [health, setHealth] = useState<Health | null>(null);
  const [connecting, setConnecting] = useState(true);
  const [model, setModel] = useState<ModelId>('simple_cnn');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [dimensions, setDimensions] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [validating, setValidating] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [comparison, setComparison] = useState<Prediction | null>(null);
  const [dragging, setDragging] = useState(false);
  const [search, setSearch] = useState('');
  const [detail, setDetail] = useState<string | null>(null);
  const [zoom, setZoom] = useState(false);
  const [showInput, setShowInput] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const selectionVersion = useRef(0);
  const selectedModel = models.find(m => m.id === model)!;
  const maxBytes = health?.max_upload_bytes || 15 * 1024 * 1024;

  async function refresh(signal?: AbortSignal) {
    try { const [h, m] = await getStatus(signal); setHealth(h); setModels(m); }
    catch { if (!signal?.aborted) { setHealth(null); setModels(old => old.map(m => ({ ...m, available: false }))); } }
    finally { if (!signal?.aborted) setConnecting(false); }
  }
  useEffect(() => {
    const controller = new AbortController(); void refresh(controller.signal);
    const timer = window.setInterval(() => void refresh(controller.signal), 30000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, []);
  useEffect(() => {
    const listener = () => { const hash = location.hash.slice(1) as Page; if (nav.some(([p]) => p === hash)) setPage(hash); };
    window.addEventListener('hashchange', listener); return () => window.removeEventListener('hashchange', listener);
  }, []);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => requestRef.current?.abort(), []);
  useEffect(() => { if (zoom && dialogRef.current && !dialogRef.current.open) dialogRef.current.showModal(); }, [zoom]);

  function navigate(next: Page) { setPage(next); location.hash = next; window.scrollTo({ top: 0, behavior: 'smooth' }); }
  async function chooseFile(next?: File) {
    if (!next || busy) return;
    const version = ++selectionVersion.current;
    setError(''); setValidating(false);
    if (!next.size) { setError('This file is empty. Choose a JPEG, PNG, or WebP image.'); return; }
    if (next.size > maxBytes) { setError(`Please choose an image smaller than ${bytes(maxBytes)}.`); return; }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type)) { setError('Please choose a JPEG, PNG, or WebP image.'); return; }
    setValidating(true);
    const url = URL.createObjectURL(next);
    try {
      const image = new Image(); image.src = url; await image.decode();
      if (version !== selectionVersion.current) return;
      setDimensions(`${image.naturalWidth} × ${image.naturalHeight}`);
      setFile(next); setResult(null); setComparison(null); setShowInput(false);
    } catch { if (version === selectionVersion.current) setError('This image could not be opened. Please choose another file.'); }
    finally { URL.revokeObjectURL(url); if (version === selectionVersion.current) setValidating(false); }
  }
  function reset() {
    ++selectionVersion.current; setValidating(false); setFile(null); setResult(null); setComparison(null); setError(''); setDimensions('');
    if (inputRef.current) inputRef.current.value = '';
    navigate('predict');
  }
  async function run(compare = false) {
    if (!file || busy || validating) return;
    const target: ModelId = compare && result ? (result.model === 'simple_cnn' ? 'mobilenetv3' : 'simple_cnn') : model;
    setBusy(true); setError('');
    const controller = new AbortController(); requestRef.current = controller;
    const timer = window.setTimeout(() => controller.abort(), 120000);
    try {
      const prediction = await predict(file, target, controller.signal);
      if (compare) setComparison(prediction);
      else { setResult(prediction); setComparison(null); window.scrollTo({ top: 0, behavior: 'smooth' }); }
    } catch (err) {
      setError(controller.signal.aborted ? 'Analysis was cancelled or took too long. You can try again.' : err instanceof Error ? err.message : 'Could not connect to the API.');
      void refresh();
    } finally { window.clearTimeout(timer); setBusy(false); requestRef.current = null; }
  }
  function downloadReport() {
    const data = new Blob([JSON.stringify({ report: result, comparison }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(data); const a = document.createElement('a'); a.href = url; a.download = `oryza-report-${result?.request_id}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#predict" onClick={() => navigate('predict')} aria-label="OryzaAI home"><span className="brand-mark"><Sprout size={29}/></span><span><strong>OryzaAI</strong><small>Precision Agritech Intelligence</small></span></a>
      <nav aria-label="Main navigation">{nav.map(([id, label]) => <a key={id} href={`#${id}`} className={page === id ? 'active' : ''} onClick={() => navigate(id)} aria-current={page === id ? 'page' : undefined}>{label}</a>)}</nav>
      <button className={`status-pill ${!health ? 'offline' : ''}`} onClick={() => void refresh()} title="Refresh API status"><span className="dot"/>{connecting ? 'Connecting to API' : health?.status === 'ok' ? 'API Online · 2 models ready' : health ? 'API Online · limited models' : 'API Offline · retry'}</button>
    </div></header>
    <main className="page-shell">
      {error && <div className="error-banner" role="alert"><CircleHelp size={19}/><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={18}/></button></div>}
      {page === 'predict' && !result && <>
        <div className="utility-bar"><span><span className={`dot ${health ? '' : 'muted'}`}/>{health ? 'FASTAPI SESSION ACTIVE' : connecting ? 'CONNECTING TO BACKEND' : 'BACKEND CONNECTION REQUIRED'}<span className="utility-separator">·</span><span className="text-neutral-600 font-normal">Research inference workspace</span></span><div><span className="tag">Local inference</span><span className="tag">Float32 · RGB</span></div></div>
        <section className="hero"><span className="eyebrow"><Microscope size={14}/> Academic research & phytopathology system</span><h1>Rice Leaf Disease Detection<br className="hidden xl:block"/> Using Deep Learning</h1><p>A focused workspace for understanding rice leaf health. Select an architecture, upload a leaf image, and explore predictions across six supported classes.</p><div className="hero-stats"><span><Leaf size={18}/><b>6</b> Supported classes</span><span><Cpu size={18}/><b>Dual</b> Deep learning architectures</span><span><ShieldCheck size={18}/><b>Private</b> Images are not retained</span></div></section>
        <div className="workspace-grid">
          <section className="space-y-4"><div className="section-label"><span>■ Foliage specimen ingestion</span><span>JPEG · PNG · WebP</span></div>
            <div className={`dropzone ${dragging ? 'dragging' : ''} ${busy ? 'disabled' : ''}`} onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void chooseFile(e.dataTransfer.files[0]); }}>
              <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" aria-label="Upload rice leaf image" disabled={busy} onChange={e => { void chooseFile(e.target.files?.[0]); e.target.value = ''; }}/>
              <span className="icon-circle"><CloudUpload size={29}/></span><h3>Drag and drop your rice leaf image here<br/>or <span className="text-emerald-700">browse files</span></h3><p>JPEG, PNG, WebP · maximum {bytes(maxBytes)}</p><div className="dropzone-meta"><span><ScanLine size={14}/> Aspect ratio preserved</span><span><ShieldCheck size={14}/> Validated before inference</span></div>
            </div>
            {file ? <div className="specimen-card"><img src={preview} alt="Selected rice leaf specimen"/><div className="min-w-0 flex-1"><div className="flex items-start justify-between gap-2"><h3 className="break-all">{file.name}</h3><span className="badge">Ready</span></div><p>Your original image will be oriented, converted to RGB, and padded to the model’s input size.</p><div className="specimen-meta"><span>{dimensions} px</span><span>{bytes(file.size)}</span><span>RGB input</span></div><div className="flex flex-wrap gap-2 mt-4"><button className="button subtle" disabled={busy} onClick={() => inputRef.current?.click()}><ImagePlus size={15}/>Change image</button><button className="button danger" disabled={busy} onClick={reset}><Trash2 size={15}/>Remove</button></div></div></div> : <div className="empty-specimen"><FileImage size={24}/><div><h3>Your specimen will appear here</h3><p>Choose a clear image with one principal leaf condition.</p></div></div>}
            <div className="upload-guide"><span className="section-label">A good image makes a difference</span><div><span><Check size={14}/> Even natural light</span><span><Check size={14}/> Sharp leaf detail</span><span><Check size={14}/> Minimal background</span></div></div>
          </section>
          <section className="space-y-3"><div className="section-label"><span>■ Neural architecture selection</span><span>Keras / TensorFlow</span></div>
            <div role="radiogroup" aria-label="Classification model" className="space-y-3">{models.map(m => <button key={m.id} role="radio" aria-checked={model === m.id} disabled={busy} className={`model-card ${model === m.id ? 'selected' : ''}`} onClick={() => { setModel(m.id); setError(''); }}><div className="flex justify-between gap-2"><div><h2>{m.name}</h2><span className="model-type">{m.id === 'simple_cnn' ? 'Trained from scratch' : 'Transfer learning'}</span></div><span className="radio-indicator">{model === m.id && <span/>}</span></div><p>{m.id === 'simple_cnn' ? 'A compact baseline with three convolution blocks, global average pooling, and a six-class classifier.' : 'A fine-tuned ImageNet backbone with depthwise convolutions and squeeze-and-excitation blocks.'}</p><div className="model-specs"><span><small>Input size</small>224 × 224 × 3</span><span><small>Normalization</small>Inside model</span><span><small>Parameters</small>{m.parameter_count ? m.parameter_count.toLocaleString() : 'Awaiting API'}</span></div><span className={`model-ready ${m.available ? '' : 'unavailable'}`}><span className="dot"/>{m.available ? 'Loaded and ready' : connecting ? 'Connecting…' : 'Currently unavailable'}</span></button>)}</div>
            <button className="analyze-button" disabled={!file || !selectedModel.available || busy || validating} onClick={() => void run()}>{busy ? <LoaderCircle className="animate-spin" size={23}/> : <Microscope size={23}/>}<span><b>{busy ? 'Analyzing your specimen…' : 'Analyze Rice Leaf'}</b><small>{busy ? 'Running the selected model' : 'Run the selected model on your image'}</small></span><ArrowRight size={21}/></button>
            <div className="flex justify-between text-[10px] tracking-wide text-emerald-900"><span>◉ Selected: {selectedModel.name}</span><span>In-memory processing</span></div>
          </section>
        </div>
        <section className="panel pipeline-panel"><div className="panel-heading"><span className="icon-circle small"><Activity size={21}/></span><div><h2>Inference Execution Pipeline</h2><p>{busy ? 'Your request is processing. Stage timings will be available when it completes.' : 'From your original image to six model scores.'}</p></div><span className="tag ml-auto">{selectedModel.name}</span></div><div className="pipeline-grid">{[['01', 'Image preparation', 'EXIF orientation · RGB · bilinear letterbox'], ['02', 'Model forward pass', 'Embedded normalization · selected architecture'], ['03', 'Class score ranking', 'Six softmax scores · top three predictions']].map(([num, title, text]) => <div key={num}><div className="flex justify-between"><small>STAGE {num}</small>{busy ? <LoaderCircle size={15} className="animate-spin"/> : <span className="stage-empty"/>}</div><h3>{title}</h3><p>{text}</p><div className={`stage-track ${busy ? 'working' : ''}`}/></div>)}</div><div className="terminal"><span>[INFERENCE]</span>{busy ? ' Request submitted · awaiting measured server results…' : file ? ` ${file.name} buffered · ready for ${selectedModel.name}` : ' Waiting for a rice leaf image…'}</div>{busy && <button className="text-button mt-3" onClick={() => requestRef.current?.abort()}>Cancel request</button>}</section>
        <section className="panel"><div className="panel-heading"><Grid2X2 size={21}/><div><h2>Supported Target Classes</h2><p>Five disease classes and one healthy reference class.</p></div><span className="tag ml-auto">6-class classifier</span></div><div className="disease-grid">{diseases.map((d, i) => <button key={d.name} className="disease-tile" onClick={() => { setDetail(d.name); navigate('library'); }}><div className="flex justify-between"><span className={`dot ${d.type === 'Healthy' ? '' : 'disease-dot'}`}/><small>CLASS 0{i+1}</small></div><h3>{d.name}</h3><p>{d.agent}</p><span>{d.type}<ChevronRight size={12}/></span></button>)}</div></section>
      </>}
      {page === 'predict' && result && <>
        <div className="report-toolbar"><div className="flex items-center gap-3"><button className="icon-button" aria-label="Return to upload" disabled={busy} onClick={() => { setResult(null); setComparison(null); }}><ArrowLeft size={19}/></button><div><p>Diagnostic Reports <span className="mx-2 text-gray-400">/</span><b>Session #{result.request_id.toUpperCase()}</b></p><small><span className="dot"/>Processed {new Date(result.created_at).toLocaleString()}</small></div></div><div className="flex flex-wrap gap-2"><button className="button subtle" disabled={busy} onClick={reset}><ImagePlus size={15}/>Analyze another</button><button className="button subtle" onClick={() => window.print()}><Download size={15}/>Export PDF</button><button className="button primary" onClick={downloadReport}><Download size={15}/>Download report</button></div></div>
        <section className="result-hero"><div><span className="badge"><span className="dot"/>Primary model prediction</span><div className="result-title"><h1>{result.predicted_class}</h1><span className="score-badge">{percent(result.prediction_score)} model score</span></div><p className="italic text-emerald-900">{diseases.find(d => d.name === result.predicted_class)?.agent}</p><p className="result-description">The highest-scoring class for this specimen. Review the full score distribution and compare architectures to explore how each model interprets your image.</p></div><div className="result-stats"><div><small>Architecture</small><b>{result.model_name}</b><span>{result.model === 'simple_cnn' ? 'Trained from scratch' : 'Fine-tuned backbone'}</span></div><div><small>Inference time</small><b>{result.timings.inference_ms.toFixed(1)} <em>ms</em></b><span>Measured on this server</span></div><div><small>Model input</small><b>224 × 224</b><span>Bilinear · RGB · padded</span></div><div><small>Score interpretation</small><b>Uncalibrated</b><span>Six-class softmax</span></div></div></section>
        <div className="report-grid">
          <div className="space-y-5"><section className="panel specimen-panel"><div className="panel-heading"><ScanLine size={21}/><h2>Specimen & Input Preview</h2><div className="segmented ml-auto"><button className={!showInput ? 'active' : ''} onClick={() => setShowInput(false)}>Original</button><button className={showInput ? 'active' : ''} onClick={() => setShowInput(true)}>224 × 224</button></div></div><div className={`report-image ${showInput ? 'letterbox' : ''}`}><img src={preview} alt={`Uploaded specimen classified as ${result.predicted_class}`}/><span className="image-caption"><span className="dot"/>{showInput ? 'Aspect-preserving preview · black padding' : 'Original specimen · whole-image classification'}</span><button className="image-zoom" aria-label="Enlarge specimen image" onClick={() => setZoom(true)}><Maximize2 size={18}/></button></div><div className="specimen-footer"><span>{result.image.width} × {result.image.height} px</span><span>{result.image.format} · {bytes(result.image.size_bytes)}</span></div><p className="footnote">Preview is illustrative; exact preprocessing is performed by Pillow on the server. This classifier does not localize lesions.</p></section>
            <section className="panel"><div className="panel-heading"><Cpu size={20}/><h2>Neural Inference Pipeline</h2><span className="tag ml-auto">TensorFlow {health?.tensorflow_version || ''}</span></div><div className="report-pipeline">{[['01 · PREPROCESS', 'RGB + letterbox', `${result.timings.preprocessing_ms.toFixed(1)} ms`], ['02 · BACKBONE', result.model_name, 'Embedded normalization'], ['03 · CLASSIFIER', '6-class softmax', `${result.timings.inference_ms.toFixed(1)} ms`]].map(([label,title,value]) => <div key={label}><small>{label}</small><h3>{title}</h3><p>{value}</p></div>)}</div><div className="tensor-details"><span><small>Input tensor</small><b>(1, 224, 224, 3)</b></span><span><small>Input range</small><b>Float32 [0, 255]</b></span><span><small>Total processing</small><b>{result.timings.total_ms.toFixed(1)} ms</b></span></div></section>
          </div>
          <div className="space-y-5"><section className="panel score-panel"><div className="panel-heading"><BarChart3 size={21}/><h2>Class Score Distribution</h2></div><p className="footnote mb-6">Relative softmax scores · not calibrated probabilities</p>{result.top_3_predictions.map((p,i) => <div className="score-row" key={p.class_name}><div><span><i className={`dot ${i ? 'muted' : ''}`}/>{i+1}. {p.class_name}</span><b>{percent(p.score)}</b></div><div className="score-track"><span style={{ width: percent(p.score) }}/></div><small>{diseases.find(d => d.name === p.class_name)?.agent}</small></div>)}<div className="remaining-scores"><small>REMAINING CLASSES</small><div>{result.all_predictions.slice(3).map(p => <div key={p.class_name}><span>{p.class_name}</span><b>{percent(p.score)}</b></div>)}</div></div></section>
            <section className="panel knowledge-panel"><div className="panel-heading"><Leaf size={21}/><h2>Botanical Knowledge Dossier</h2></div><div className="knowledge-block"><h3><Microscope size={16}/>About this class</h3><p>{diseases.find(d => d.name === result.predicted_class)?.description}</p></div><div className="knowledge-block"><h3><ScanLine size={16}/>What this result can tell you</h3><p>The model found this class most similar to its learned examples. It does not confirm a pathogen, estimate severity, or detect unknown conditions.</p></div><div className="knowledge-block"><h3><ShieldCheck size={16}/>A useful next step</h3><p>Review the original leaf, collect additional clear photographs, and discuss the symptoms with a local agricultural extension officer.</p></div><a className="source-link" href={diseaseSource} target="_blank" rel="noreferrer">Explore IRRI’s rice disease reference <ExternalLink size={14}/></a></section>
          </div>
        </div>
        <section className="compare-banner"><span className="icon-circle"><Activity size={23}/></span><div><h3>Compare with {result.model === 'simple_cnn' ? 'MobileNetV3Large' : 'Simple CNN'}</h3><p>Run the same specimen through the other architecture.</p></div><button className="button subtle ml-auto" disabled={busy || !models.find(m => m.id !== result.model)?.available} onClick={() => void run(true)}>{busy ? <LoaderCircle size={17} className="animate-spin"/> : <Cpu size={17}/>} {busy ? 'Comparing…' : comparison ? 'Run comparison again' : 'Compare models'}</button></section>
        {comparison && <section className="panel comparison-results" aria-live="polite"><div className="panel-heading"><BarChart3 size={20}/><h2>Same-image Model Comparison</h2><span className="badge ml-auto">{result.predicted_class === comparison.predicted_class ? 'Models agree' : 'Models disagree'}</span></div><div className="grid sm:grid-cols-2 gap-4">{[result,comparison].map(r => <div className="knowledge-block" key={r.model}><h3>{r.model_name}</h3><b className="text-xl">{r.predicted_class}</b><p>{percent(r.prediction_score)} model score · {r.timings.inference_ms.toFixed(1)} ms</p></div>)}</div><p className="footnote mt-3">Agreement does not establish correctness. Scores from independently trained models are not calibrated against each other.</p></section>}
        <Notice/><div className="report-bottom"><p><Grid2X2 size={17}/>Ready to explore another specimen?</p><div className="flex gap-2"><button className="button subtle" onClick={() => navigate('library')}>Browse disease library</button><button className="button primary" disabled={busy} onClick={reset}><ImagePlus size={16}/>Analyze another specimen</button></div></div>
      </>}
      {page === 'library' && <>
        <section className="page-intro"><span className="overline"><BookOpen size={16}/>The reference collection</span><h1>Disease Library</h1><p>Explore the six labels your models were trained to recognize.</p></section><div className="library-toolbar"><label className="search-field"><Search size={18}/><input placeholder="Search classes or pathogens…" aria-label="Search disease library" value={search} onChange={e => setSearch(e.target.value)}/></label><span className="tag">5 disease classes + 1 healthy class</span></div><div className="library-grid">{diseases.filter(d => `${d.name} ${d.agent}`.toLowerCase().includes(search.toLowerCase())).map(d => <article key={d.name} className={`panel library-card ${detail === d.name ? 'highlight' : ''}`}><div className="flex items-center justify-between mb-6"><span className="icon-circle">{d.type === 'Healthy' ? <Sprout size={25}/> : <Leaf size={25}/>}</span><span className="tag">{d.type}</span></div><small className="overline">CLASS {String(diseases.indexOf(d)+1).padStart(2,'0')}</small><h2>{d.name}</h2><p className="italic">{d.agent}</p><p>{d.description}</p><button className="text-button" onClick={() => setDetail(detail === d.name ? null : d.name)}>{detail === d.name ? 'Hide model context' : 'View model context'}<ChevronRight size={15}/></button>{detail === d.name && <div className="knowledge-block mt-4"><p>Both architectures include this label. The API preserves the class order exported with each model. Upload a specimen to view its score for this class alongside all five alternatives.</p><button className="text-button mt-3" onClick={() => navigate('predict')}>Analyze a specimen <ArrowRight size={15}/></button></div>}</article>)}</div>{!diseases.some(d => `${d.name} ${d.agent}`.toLowerCase().includes(search.toLowerCase())) && <div className="empty-state"><Search size={30}/><h2>No matching classes</h2><p>Try a disease name such as “Brown Spot”.</p></div>}<a className="source-link my-6" href={diseaseSource} target="_blank" rel="noreferrer">Botanical reference: IRRI Rice Diseases Online Resource <ExternalLink size={14}/></a><Notice/>
      </>}
      {page === 'benchmark' && <>
        <section className="page-intro"><span className="overline"><BarChart3 size={16}/>Evaluation evidence</span><h1>Model Benchmark</h1><p>Internal held-out results from your supplied model packages.</p></section><div className="grid md:grid-cols-2 gap-6 mb-6">{models.map(m => <section className="panel benchmark-card" key={m.id}><span className="tag">{m.id === 'simple_cnn' ? 'Baseline architecture' : 'Transfer learning'}</span><h2>{m.name}</h2><p>{m.description}</p><div className="benchmark-value">{m.metrics ? percent(m.metrics.accuracy) : '—'}<span>Test accuracy</span></div><div className="benchmark-bar"><span style={{ width: m.metrics ? percent(m.metrics.accuracy) : '0%' }}/></div><div className="benchmark-metrics"><span><small>Macro F1</small><b>{m.metrics ? m.metrics.macro_f1.toFixed(4) : 'Unavailable'}</b></span><span><small>Test images</small><b>{m.metrics?.images ?? '—'}</b></span><span><small>Parameters</small><b>{m.parameter_count?.toLocaleString() ?? '—'}</b></span><span><small>Model size</small><b>{m.size_bytes ? bytes(m.size_bytes) : '—'}</b></span></div><p className="footnote break-all">Model SHA256: {m.sha256 || 'Awaiting API'}</p></section>)}</div><section className="panel"><div className="panel-heading"><FlaskConical size={21}/><h2>Read the results in context</h2></div><p className="body-copy">These metrics are imported from each package’s test_metrics.json. They describe internal held-out evaluations, not independent field performance. {models[0]?.split_fingerprint && models[0].split_fingerprint === models[1]?.split_fingerprint ? 'The exported manifests have matching image identities, labels, and train/validation/test assignments. Both test evaluations use the same 734 images.' : 'Matching split manifests have not been verified for the currently loaded packages.'} Use the same uploaded image to compare the models’ predictions directly.</p><button className="button primary mt-5" onClick={() => navigate('predict')}>Compare your own specimen<ArrowRight size={16}/></button></section><Notice/>
      </>}
      {page === 'docs' && <>
        <section className="page-intro"><span className="overline"><BookOpen size={16}/>A transparent research workflow</span><h1>System Documentation</h1><p>How your image becomes a model prediction.</p></section><div className="docs-grid"><section className="panel"><h2>From upload to result</h2><ol className="docs-steps"><li><b>Choose an image and a model.</b><p>Use a still JPEG, PNG, or WebP image up to {bytes(maxBytes)}. Only the selected model runs.</p></li><li><b>Validate and prepare.</b><p>The backend checks actual image content, applies EXIF orientation, composites transparency on black, converts to RGB, and fits the image onto a 224 × 224 black canvas using bilinear resizing.</p></li><li><b>Run inference.</b><p>The tensor is float32 in [0, 255]. Each saved model applies its own normalization internally. Both models load once at startup.</p></li><li><b>Inspect the scores.</b><p>All six scores and the top three classes are returned with measured processing times. Uploaded images are not retained by the application.</p></li></ol></section><section className="panel"><h2>REST API</h2><div className="endpoint"><code>GET</code><b>/api/health</b><p>Backend health, model availability, and upload limit.</p></div><div className="endpoint"><code>GET</code><b>/api/models</b><p>Model metadata, preprocessing, classes, and evaluation metrics.</p></div><div className="endpoint"><code>POST</code><b>/api/predict</b><p>Multipart fields: <code>file</code> and <code>model</code> (simple_cnn or mobilenetv3).</p></div><a className="button primary mt-5" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">Open interactive API docs<ExternalLink size={16}/></a></section></div><section className="panel mt-6"><h2>Report exports & reproducibility</h2><p className="body-copy mt-3">Use Export PDF on the results screen and choose “Save as PDF” in your browser’s print dialog. Download report saves the original JSON prediction, model checksum, timestamp, timings, and any comparison result. Reports remain on your device; no public sharing link or database is created.</p></section><Notice/>
      </>}
    </main>
    <footer><div className="footer-inner"><div><h3>OryzaAI Research Initiative <span className="tag">v1.0</span></h3><p>Rice leaf classification for academic research.<br/>Built with your trained models. Designed for careful interpretation.</p></div><div className="footer-links"><a href="#docs" onClick={() => navigate('docs')}>Documentation & API</a><a href="#benchmark" onClick={() => navigate('benchmark')}>Validation metrics</a><a href={diseaseSource} target="_blank" rel="noreferrer">Botanical reference<ExternalLink size={12}/></a></div></div><div className="footer-bottom"><span>OryzaAI · Precision agriculture research</span><span>React + Tailwind CSS · FastAPI + TensorFlow</span></div></footer>
    {zoom && <dialog ref={dialogRef} className="image-modal" aria-label="Enlarged specimen" onCancel={() => setZoom(false)}><button autoFocus className="icon-button" aria-label="Close enlarged image" onClick={() => setZoom(false)}><X/></button><img src={preview} alt="Enlarged uploaded specimen"/></dialog>}
  </>;
}
