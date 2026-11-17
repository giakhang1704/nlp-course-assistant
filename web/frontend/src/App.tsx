import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { ArrowUp, Plus, BookOpen, PanelLeftClose, PanelLeftOpen, ChevronDown, MoreHorizontal, Pencil, Trash2, X, Copy, Check, Download, RotateCcw, FileText, Code2, CalendarDays, GraduationCap, ListChecks, ArrowUpRight, LoaderCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from './api'
import { ParseAnalysis } from './ParseAnalysis'
import type { Conversation, ConversationFull, Course, Turn } from './api'

type Modal = 'search' | 'chapters' | 'settings' | 'rename' | 'delete' | 'reset' | 'sources' | 'details' | null
type Theme = 'light' | 'dark' | 'system'

function Dialog({ title, onClose, children, drawer = false }: { title: string; onClose: () => void; children: ReactNode; drawer?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => { const el = ref.current!; el.showModal(); return () => el.close() }, [])
  return <dialog ref={ref} className={drawer ? 'dialog drawer' : 'dialog'} onCancel={e => { e.preventDefault(); onClose() }} onClick={e => { const r = e.currentTarget.getBoundingClientRect(); if (e.target === e.currentTarget && (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)) onClose() }}>
    <div className="dialog-heading"><h2>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Đóng"><X /></button></div>{children}
  </dialog>
}

function Orb({ small = false }: { small?: boolean }) { return <span className={`orb${small ? ' small' : ''}`} aria-hidden="true"><span /></span> }

function Composer({ value, onChange, onSend, busy, ready, onChapters }: { value: string; onChange: (v: string) => void; onSend: () => void; busy: boolean; ready: boolean; onChapters: () => void }) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { const el = ref.current!; el.style.height = '56px'; el.style.height = `${Math.min(el.scrollHeight, 180)}px` }, [value])
  function submit(e: FormEvent) { e.preventDefault(); if (value.trim() && !busy && ready) onSend() }
  return <form className="composer" onSubmit={submit}>
    <textarea ref={ref} aria-label="Câu hỏi" placeholder="Bạn muốn hỏi điều gì?" value={value} maxLength={4000} onChange={e => onChange(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (value.trim() && !busy && ready) onSend() } }} />
    <div className="composer-toolbar"><button type="button" className="scope-button" onClick={onChapters}><BookOpen /><span>Môn NLP</span><ChevronDown /></button><button className="send-button" aria-label="Gửi câu hỏi" disabled={!value.trim() || busy || !ready}>{busy ? <LoaderCircle className="spin" /> : <ArrowUp />}</button></div>
  </form>
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<Record<string, ConversationFull>>({})
  const [course, setCourse] = useState<Course | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<{id: string; question: string} | null>(null)
  const [loading, setLoading] = useState(false)
  const [modal, setModal] = useState<Modal>(null)
  const [target, setTarget] = useState<Conversation | null>(null)
  const [sourceTurn, setSourceTurn] = useState<Turn | null>(null)
  const [rename, setRename] = useState('')
  const [copied, setCopied] = useState('')
  const [sidebar, setSidebar] = useState(() => window.innerWidth > 800)
  const [menu, setMenu] = useState<string | null>(null)
  const [theme] = useState<Theme>(() => (localStorage.getItem('ca-theme') as Theme) || 'light')
  const [fontSize] = useState(() => localStorage.getItem('ca-font-size') || 'normal')
  const [showDetails] = useState(() => localStorage.getItem('ca-details') !== 'false')
  const tail = useRef<HTMLDivElement>(null)
  const cache = useRef(loaded)
  const initialized = useRef(false)
  useEffect(() => { cache.current = loaded }, [loaded])
  const active = selected ? loaded[selected] : null
  const turns = active?.turns || []
  const pending = busy?.id === selected ? busy : null
  const draftKey = selected || 'new'
  const draft = drafts[draftKey] || ''

  async function initialize() {
    setError(''); setReady(false)
    try {
      const sessions = await api<Conversation[]>('/conversations')
      setConversations(sessions)
      const [health, information] = await Promise.all([api<{ready: boolean; error: string | null}>('/health'), api<Course>('/course')])
      if (!health.ready) throw new Error(health.error || 'Đang tải dữ liệu môn học.')
      setCourse(information); setReady(true)
      const last = localStorage.getItem('ca-active')
      if (last && sessions.some(c => c.id === last)) setSelected(last)
    } catch (e) { setError((e as Error).message) }
  }
  useEffect(() => { if (!initialized.current) { initialized.current = true; void initialize() } }, [])
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => { document.documentElement.dataset.theme = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme }
    apply(); localStorage.setItem('ca-theme', theme); media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [theme])
  useEffect(() => { localStorage.setItem('ca-font-size', fontSize); document.documentElement.dataset.fontSize = fontSize }, [fontSize])
  useEffect(() => { localStorage.setItem('ca-details', String(showDetails)) }, [showDetails])
  useEffect(() => {
    if (!selected) return
    localStorage.setItem('ca-active', selected)
    if (cache.current[selected]) return
    let ignore = false
    setLoading(true)
    api<ConversationFull>(`/conversations/${selected}`).then(c => { if (!ignore) setLoaded(old => ({...old, [c.id]: c})) }).catch(e => { if (!ignore) setError(e.message) }).finally(() => { if (!ignore) setLoading(false) })
    return () => { ignore = true; setLoading(false) }
  }, [selected])
  useEffect(() => { tail.current?.scrollIntoView({behavior: 'smooth', block: 'end'}) }, [selected, turns.length, pending?.question])
  useEffect(() => { function key(e: KeyboardEvent) { if (e.key === 'Escape') setMenu(null) } window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key) }, [])
  useEffect(() => { if (!menu) return; const close = () => setMenu(null); window.addEventListener('click', close); return () => window.removeEventListener('click', close) }, [menu])

  function newChat() { localStorage.removeItem('ca-active'); setSelected(null); setError(''); if (window.innerWidth <= 800) setSidebar(false) }
  function openChat(id: string) { setSelected(id); setError(''); setModal(null); if (window.innerWidth <= 800) setSidebar(false) }
  async function send(text = draft) {
    const content = text.trim()
    if (!content || busy || !ready) return
    let id = selected
    const originalDraft = draftKey
    setError('')
    // Reserve the operation immediately, including while the conversation is created.
    setBusy({id: id || 'creating', question: content})
    try {
      if (!id) {
        const conv = await api<Conversation>('/conversations', {method: 'POST'})
        id = conv.id
        setLoaded(old => ({...old, [conv.id]: {...conv, turns: []}})); setConversations(old => [conv, ...old]); setSelected(id)
      }
      const targetId = id
      setBusy({id: targetId, question: content})
      setDrafts(old => ({...old, [originalDraft]: '', [targetId]: ''}))
      const turn = await api<Turn>(`/conversations/${targetId}/messages`, {method: 'POST', body: JSON.stringify({content, request_id: crypto.randomUUID()})})
      setLoaded(old => ({...old, [targetId]: {...old[targetId], turns: [...(old[targetId]?.turns || []), turn]}}))
      setConversations(await api<Conversation[]>('/conversations'))
    } catch (e) { setError((e as Error).message); setDrafts(old => ({...old, [id || 'new']: content})) }
    finally { setBusy(null) }
  }
  async function changeConversation(action: 'rename' | 'delete' | 'reset') {
    const conv = target || conversations.find(c => c.id === selected)
    if (!conv || busy) return
    try {
      if (action === 'rename') await api(`/conversations/${conv.id}`, {method: 'PATCH', body: JSON.stringify({title: rename})})
      else await api(`/conversations/${conv.id}${action === 'reset' ? '/reset' : ''}`, {method: action === 'delete' ? 'DELETE' : 'POST'})
      setLoaded(old => { const next = {...old}; delete next[conv.id]; return next })
      setConversations(await api<Conversation[]>('/conversations'))
      if (action !== 'rename' && conv.id === selected) newChat()
      setModal(null)
    } catch (e) { setError((e as Error).message) }
  }
  async function copy(turn: Turn) { try { await navigator.clipboard.writeText(turn.content); setCopied(turn.id); window.setTimeout(() => setCopied(''), 1800) } catch { setError('Không sao chép được nội dung.') } }
  function download() {
    if (!active?.turns.length) return
    const content = `# ${conversations.find(c => c.id === selected)?.title || 'Course Assistant'}\n\n` + turns.map(t => `## ${t.question}\n\n${t.content}\n` + (t.sources.length ? `\n### Nguồn\n${t.sources.map(s => `- ${s.title} — ${s.file}`).join('\n')}\n` : '')).join('\n')
    const url = URL.createObjectURL(new Blob([content], {type: 'text/markdown;charset=utf-8'})); const a = document.createElement('a'); a.href = url; a.download = 'hoi-thoai.md'; a.click(); URL.revokeObjectURL(url)
  }
  const suggestions = [
    {icon: GraduationCap, label: 'Thông tin môn học', question: 'Môn NLP có bao nhiêu tín chỉ?'},
    {icon: BookOpen, label: 'Nội dung bài học', question: ''},
    {icon: CalendarDays, label: 'Lịch học', question: 'Tuần 5 học những gì?'},
    {icon: ListChecks, label: 'Bài tập lớn', question: 'Bài tập lớn có những yêu cầu gì?'},
  ]
  const empty = !turns.length && !pending && !loading

  return <div className={`app-shell${sidebar ? ' sidebar-visible' : ''}`}>
    {sidebar && <button className="mobile-scrim" aria-label="Đóng thanh bên" onClick={() => setSidebar(false)} />}
    <aside className="sidebar" aria-label="Thanh điều hướng">
      <div className="brand-row"><button className="brand" onClick={newChat}><Orb /><span>Course Assistant</span></button><button className="icon-button sidebar-toggle" onClick={() => setSidebar(false)} title="Thu gọn thanh bên" aria-label="Thu gọn thanh bên"><PanelLeftClose /></button></div>
      <button className="new-chat" onClick={newChat}><Plus /><span>Cuộc trò chuyện mới</span></button>
      <nav className="main-nav"><button onClick={() => setModal('chapters')}><BookOpen /><span>Nội dung môn học</span></button></nav>
      <div className="history-heading">Hội thoại</div>
      <div className="chat-history">{conversations.map(conv => <div className={`history-item${conv.id === selected ? ' active' : ''}`} key={conv.id}>
        <button className="history-title" onClick={() => openChat(conv.id)} title={conv.title}>{conv.title}</button>
        <button className="history-more icon-button" aria-label={`Tùy chọn ${conv.title}`} onClick={e => {e.stopPropagation(); setMenu(menu === conv.id ? null : conv.id)}}><MoreHorizontal /></button>
        {menu === conv.id && <div className="context-menu" role="menu"><button role="menuitem" onClick={() => {setTarget(conv); setRename(conv.title); setModal('rename')}}><Pencil />Đổi tên</button><button role="menuitem" className="danger-text" disabled={!!busy} onClick={() => {setTarget(conv); setModal('delete')}}><Trash2 />Xóa hội thoại</button></div>}
      </div>)}</div>
    </aside>

    <main className="main-panel">
      <header className="topbar"><div className="topbar-left">{!sidebar && <button className="icon-button" onClick={() => setSidebar(true)} aria-label="Mở thanh bên"><PanelLeftOpen /></button>}<button className="course-selector" onClick={() => setModal('chapters')}>CO3085 <ChevronDown /></button></div><div className="topbar-right">{!!turns.length && <><button className="icon-button" title="Xuất hội thoại" aria-label="Xuất hội thoại" onClick={download}><Download /></button><button className="icon-button" disabled={!!busy} title="Bắt đầu lại ngữ cảnh" aria-label="Bắt đầu lại ngữ cảnh" onClick={() => {setTarget(null); setModal('reset')}}><RotateCcw /></button></>}<span className="profile-avatar" title="Không gian học tập">K</span></div></header>
      {error && <div className="error-banner" role="alert"><span>{error}</span>{!ready && <button onClick={() => void initialize()}>Thử lại</button>}<button className="icon-button" aria-label="Đóng thông báo" onClick={() => setError('')}><X /></button></div>}
      <div className={`conversation-area${empty ? ' is-empty' : ''}`}>
        {empty ? <section className="welcome"><div className="welcome-emblem"><Orb /></div><h1><em>Chào bạn,</em><span>Hôm nay bạn muốn tìm hiểu gì?</span></h1><Composer value={draft} onChange={v => setDrafts(old => ({...old, [draftKey]: v}))} onSend={() => void send()} busy={!!busy} ready={ready} onChapters={() => setModal('chapters')} /><div className="suggestion-row">{suggestions.map(item => <button key={item.label} disabled={!!busy || !ready} onClick={() => item.question ? void send(item.question) : setModal('chapters')}><item.icon /><span>{item.label}</span></button>)}</div></section> : <div className="messages" role="log" aria-live="polite" aria-label="Nội dung hội thoại">
          {loading && <div className="loading-history"><LoaderCircle className="spin" aria-label="Đang tải hội thoại" /></div>}
          {turns.map(turn => <section className="turn" key={turn.id}><div className="user-message">{turn.question}</div><ParseAnalysis turn={turn}/><div className={`assistant-message${turn.error ? ' message-error' : ''}`}><div className="assistant-avatar"><Orb small /></div><div className="assistant-content"><div className="prose"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{a: ({...props}) => <a {...props} target="_blank" rel="noopener noreferrer" />}}>{turn.content}</ReactMarkdown></div><div className="message-actions"><button className="icon-button" title="Sao chép" aria-label="Sao chép câu trả lời" onClick={() => void copy(turn)}>{copied === turn.id ? <Check /> : <Copy />}</button>{turn.sources.length > 0 && <button className="source-button" onClick={() => {setSourceTurn(turn); setModal('sources')}}><FileText /><span>Nguồn</span><span className="source-count">{turn.sources.length}</span></button>}{showDetails && <button className="icon-button" title="Chi tiết retrieval và ngữ cảnh" aria-label="Chi tiết retrieval và ngữ cảnh" onClick={() => {setSourceTurn(turn); setModal('details')}}><Code2 /></button>}</div></div></div></section>)}
          {pending && <section className="turn pending-turn"><div className="user-message">{pending.question}</div><div className="assistant-message"><div className="assistant-avatar"><Orb small /></div><div className="thinking" role="status" aria-label="Đang trả lời"><i /><i /><i /></div></div></section>}
          <div ref={tail} />
        </div>}
      </div>
      {!empty && <div className="composer-dock"><Composer value={draft} onChange={v => setDrafts(old => ({...old, [draftKey]: v}))} onSend={() => void send()} busy={!!busy} ready={ready} onChapters={() => setModal('chapters')} /></div>}
    </main>

    {modal === 'chapters' && <Dialog title="Nội dung môn học" drawer onClose={() => setModal(null)}><div className="course-summary"><span className="course-icon"><GraduationCap /></span><div><h3>Xử lý Ngôn ngữ Tự nhiên</h3><span>CO3085</span></div></div><div className="chapter-list">{course?.chapters.map(ch => <button key={ch.number} disabled={!!busy} onClick={() => {setModal(null); void send(`Chương ${ch.number} nói về gì?`)}}><span className="chapter-number">{String(ch.number).padStart(2, '0')}</span><span>{ch.title}</span><ArrowUpRight /></button>)}</div></Dialog>}
    {(modal === 'rename' || modal === 'delete' || modal === 'reset') && <Dialog title={modal === 'rename' ? 'Đổi tên hội thoại' : modal === 'delete' ? 'Xóa hội thoại?' : 'Bắt đầu lại?'} onClose={() => setModal(null)}><form className="action-form" onSubmit={e => {e.preventDefault(); void changeConversation(modal as 'rename' | 'delete' | 'reset')}}>{modal === 'rename' ? <input autoFocus aria-label="Tên hội thoại" maxLength={100} value={rename} onChange={e => setRename(e.target.value)} /> : <p>{modal === 'delete' ? 'Cuộc trò chuyện và ngữ cảnh sẽ được xóa.' : 'Xóa tin nhắn và ngữ cảnh của cuộc trò chuyện này.'}</p>}<div className="dialog-actions"><button type="button" className="button secondary" onClick={() => setModal(null)}>Hủy</button><button className={`button${modal === 'delete' ? ' destructive' : ''}`} disabled={!!busy || (modal === 'rename' && !rename.trim())}>{modal === 'rename' ? 'Lưu' : modal === 'delete' ? 'Xóa' : 'Bắt đầu lại'}</button></div></form></Dialog>}
    {modal === 'sources' && sourceTurn && <Dialog title="Nguồn tham khảo" drawer onClose={() => setModal(null)}><div className="source-list">{sourceTurn.sources.map((source, index) => <article className="source-card" key={source.label}><div className="source-card-heading"><span className="source-index">{index + 1}</span><h3>{source.title}</h3></div><div className="source-meta">{source.chapter && <span>Chương {source.chapter}</span>}{source.is_sample && <span className="sample-tag">Dữ liệu mẫu</span>}{source.slides.length > 0 && <span>Slide {source.slides.join(', ')}</span>}</div><div className="source-file"><FileText /><span>{source.file.split('/').pop()}</span></div><p className="source-excerpt">{source.text}</p></article>)}{sourceTurn.has_sample_source && !sourceTurn.sources.some(s => s.is_sample) && <span className="sample-tag">Có sử dụng ngữ cảnh từ lịch hoặc quy định mẫu</span>}</div></Dialog>}
    {modal === 'details' && sourceTurn && <Dialog title="Phân tích câu hỏi" drawer onClose={() => setModal(null)}><div className="details-body"><h3>Câu đã diễn giải</h3><p>{sourceTurn.details.effective_question}</p><h3>Cây cú pháp</h3><pre>{sourceTurn.details.parse_tree}</pre>{sourceTurn.details.context_used.length > 0 && <><h3>Ngữ cảnh được sử dụng</h3><pre>{JSON.stringify(sourceTurn.details.context_used, null, 2)}</pre></>}{sourceTurn.details.retrieval.map((r, i) => <details key={i}><summary>{r.title}</summary><p>{r.text}</p></details>)}</div></Dialog>}
  </div>
}

export default App
