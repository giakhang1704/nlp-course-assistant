import { useMemo, useRef, useState } from 'react'
import { ChevronRight, GitBranch, Maximize2, Minus, Plus, X } from 'lucide-react'
import type { Analysis, Turn } from './api'
import { compactTree, layoutTree, readTree, treeRules } from './parseTree'
import './parse-analysis.css'

function TreeView({ source }: { source: string }) {
  const [zoom, setZoom] = useState(0.75)
  const [compact, setCompact] = useState(true)
  const tree = useMemo(() => readTree(source), [source])
  const drawing = useMemo(() => tree ? layoutTree(compact ? compactTree(tree) : tree) : null, [tree, compact])
  const viewport = useRef<HTMLDivElement>(null)
  const drag = useRef<{x: number; y: number; left: number; top: number} | null>(null)
  if (!drawing) return <pre className="tree-raw">{source}</pre>
  return <div className="tree-view">
    <div className="tree-toolbar"><span>Cây cú pháp</span><div className="tree-controls"><div className="tree-density" role="group" aria-label="Mức chi tiết cây"><button aria-pressed={compact} title="Gộp các terminal liền nhau cùng nhánh" onClick={() => setCompact(true)}>Gọn</button><button aria-pressed={!compact} title="Hiện từng token của parser" onClick={() => setCompact(false)}>Đầy đủ</button></div><button aria-label="Thu nhỏ cây" disabled={zoom <= .3} onClick={() => setZoom(z => Math.max(.3, z - .15))}><Minus /></button><button className="tree-zoom" aria-label="Đặt lại kích thước cây" onClick={() => setZoom(.75)}>{Math.round(zoom * 100)}%</button><button aria-label="Phóng to cây" disabled={zoom >= 1.8} onClick={() => setZoom(z => Math.min(1.8, z + .15))}><Plus /></button></div></div>
    <div ref={viewport} className="tree-viewport" tabIndex={0} aria-label="Cây cú pháp, cuộn hoặc kéo để di chuyển" onPointerDown={e => {
      if (e.pointerType !== 'mouse' || e.button !== 0) return
      const el = e.currentTarget
      drag.current = {x:e.clientX,y:e.clientY,left:el.scrollLeft,top:el.scrollTop}; el.setPointerCapture(e.pointerId)
    }} onPointerMove={e => { if (drag.current) {e.currentTarget.scrollLeft = drag.current.left + drag.current.x - e.clientX; e.currentTarget.scrollTop = drag.current.top + drag.current.y - e.clientY} }} onPointerUp={() => {drag.current = null}} onLostPointerCapture={() => {drag.current = null}}>
      <svg role="img" aria-label="Cây cú pháp từ parser Earley" style={{width:drawing.width * zoom,height:drawing.height * zoom}} viewBox={`0 0 ${drawing.width} ${drawing.height}`}>
        <title>{compact ? "Cây cú pháp rút gọn: gộp terminal liền nhau cùng nhánh" : "Cây cú pháp đầy đủ từ parser Earley"}</title>
        {drawing.edges.map((e,i) => <path key={i} className="tree-edge" d={`M ${e.x1} ${e.y1} C ${e.x1} ${(e.y1+e.y2)/2}, ${e.x2} ${(e.y1+e.y2)/2}, ${e.x2} ${e.y2}`} />)}
        {drawing.nodes.map(({node,x,y,width,id}) => <g key={id} className={node.terminal ? 'tree-node terminal' : 'tree-node'}><title>{node.label}{!node.terminal && !node.children.length ? ' → ε' : ''}</title><rect x={x-width/2} y={y-18} width={width} height={36} rx={8}/><text x={x} y={y} textAnchor="middle" dominantBaseline="central">{node.label}{!node.terminal && !node.children.length ? ' · ε' : ''}</text></g>)}
      </svg>
    </div>
  </div>
}

function AnalysisBody({ analysis, fallback }: { analysis?: Analysis | null; fallback: string }) {
  const source = analysis?.parse_tree ?? fallback
  const tree = useMemo(() => readTree(source), [source])
  const rules = useMemo(() => tree ? treeRules(tree) : [], [tree])
  const frame = analysis?.semantic_frame
  return <div className="parse-body">
    <div className="parse-summary"><span className={`parse-badge${tree ? ' valid' : ''}`}>{tree ? 'Thuộc văn phạm' : source.trim() === '()' ? 'Chưa thuộc văn phạm' : 'Không đọc được cây'}</span><span>Earley · CFG</span></div>
    {analysis?.normalized_sentence && <div className="parse-sentence">{analysis.normalized_sentence}</div>}
    {tree ? <TreeView source={source}/> : <div className="parse-empty"><code>{source}</code><p>{source.trim() === '()' ? 'Câu chưa được văn phạm hiện tại bao phủ.' : 'Xem biểu diễn gốc bên dưới.'}</p></div>}
    {!!analysis?.ambiguity && <p className="parse-warning">Có nhiều cách diễn giải: {analysis.candidate_intents.join(', ')}. Cây hiển thị là một kết quả phân tích.</p>}
    {rules.length > 0 && <details className="parse-section"><summary>Grammar rules <span>{rules.length}</span></summary><div className="parse-rules">{rules.map(rule => <code key={rule}>{rule}</code>)}</div></details>}
    {analysis && <details className="parse-section"><summary>Diễn giải ngữ nghĩa</summary><dl className="parse-fields"><dt>Intent</dt><dd>{analysis.intent || '—'}</dd>{['attribute','operation','question_type'].map(key => frame?.[key] ? <div className="parse-field" key={key}><dt>{key}</dt><dd>{String(frame[key])}</dd></div> : null)}<dt>Entities</dt><dd>{analysis.entities.length ? analysis.entities.map((entity,i) => <span className="parse-entity" key={i}>{entity.surface || entity.value || entity.id} <small>{entity.type}{entity.id ? ` · ${entity.id}` : ''}</small></span>) : '—'}</dd></dl></details>}
    <details className="parse-section"><summary>Biểu diễn cây gốc</summary><pre className="tree-raw">{source}</pre></details>
  </div>
}

export function ParseAnalysis({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'original' | 'effective'>('original')
  const dialog = useRef<HTMLDialogElement>(null)
  // Older saved conversations only have the bracketed tree. They remain viewable.
  const details = turn.details as Turn['details'] & {analysis?: Analysis; effective_analysis?: Analysis | null}
  const analysis = mode === 'effective' ? details.effective_analysis : details.analysis
  const hasEffective = !!details.effective_analysis && details.effective_question !== turn.question
  function body() { return <>
    {hasEffective && <div className="parse-tabs" role="group" aria-label="Câu được phân tích"><button aria-pressed={mode === 'original'} onClick={() => setMode('original')}>Câu gốc</button><button aria-pressed={mode === 'effective'} onClick={() => setMode('effective')}>Sau diễn giải ngữ cảnh</button></div>}
    {mode === 'effective' && <p className="parse-effective">{details.effective_question}</p>}
    <AnalysisBody analysis={analysis} fallback={details.parse_tree}/>
  </> }
  return <div className="parse-analysis">
    <button className="parse-toggle" aria-expanded={open} onClick={() => setOpen(!open)}><ChevronRight className={open ? 'expanded' : ''}/><GitBranch/>Phân tích câu</button>
    {open && <div className="parse-panel"><div className="parse-panel-heading"><span>Grammar & Parser</span><button className="icon-button" aria-label="Mở rộng phân tích câu" onClick={() => dialog.current?.showModal()}><Maximize2/></button></div>{body()}</div>}
    <dialog ref={dialog} className="dialog parse-dialog"><div className="dialog-heading"><h2>Phân tích câu</h2><button className="icon-button" aria-label="Đóng phân tích câu" onClick={() => dialog.current?.close()}><X/></button></div><p className="parse-question">{turn.question}</p>{body()}</dialog>
  </div>
}

