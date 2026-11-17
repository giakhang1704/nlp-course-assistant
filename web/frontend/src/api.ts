export type Source = { label: string; title: string; file: string; chapter?: number; slides: number[]; is_sample: boolean; text: string }
export type Analysis = { sentence: string; normalized_sentence: string; status: string; is_valid: boolean; parse_tree: string; intent: string | null; entities: {type: string; surface?: string; value?: string; id?: string}[]; semantic_frame: Record<string, unknown> | null; ambiguity: boolean; candidate_intents: string[] }
export type Turn = { id: string; question: string; content: string; status: string; created_at: string; sources: Source[]; has_sample_source: boolean; error: boolean; details: { original_question: string; effective_question: string; parse_tree: string; intent: string | null; context_used: unknown[]; retrieval: {title: string; score: number; text: string}[] } }
export type Conversation = { id: string; title: string; created_at: string; updated_at: string }
export type ConversationFull = Conversation & { turns: Turn[] }
export type Course = { id: string; name: string; chapters: {number: number; title: string}[] }

const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${base}/api${path}`, { credentials: 'include', ...options, headers: { 'Content-Type': 'application/json', ...options.headers } })
  const value = await response.json().catch(() => null)
  if (!response.ok) throw new Error(typeof value?.detail === 'string' ? value.detail : 'Không thể kết nối. Hãy thử lại.')
  return value as T
}
