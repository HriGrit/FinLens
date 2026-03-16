import { create } from 'zustand'

export interface Citation {
  index: number
  company: string | null
  year: string | null
  doc_type: string | null
  page_number: number | null
  filename: string | null
  excerpt?: string
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd?: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  trace_id?: string
  citations?: Citation[]
  reasoning?: ReasoningPayload
  usage?: TokenUsage
  model?: string
  latency_ms?: number
}

export interface ReasoningPayload {
  summary: string
  retrieval: {
    company_filter?: string
    year_filter?: string
    retrieval_top_k: number
    retrieved_candidates: number
    latency_ms: number
  }
  rerank: {
    requested_rerank_top_k: number
    selected_nodes: number
    top_sources: Array<{
      filename: string | null
      page_number: number | null
      company: string | null
      year: string | null
    }>
  }
  generation: {
    model: string
    latency_ms: number
    prompt_tokens: number | null
    completion_tokens: number | null
    total_tokens: number | null
    cost_usd?: number | null
  }
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  usage: TokenUsage
  model: string
  latency_ms: number
  trace_id: string
  reasoning: ReasoningPayload
}

export interface FreeModelOption {
  id: string
  name: string
  context_length?: number
  description?: string
}

export const DEFAULT_FREE_MODELS: FreeModelOption[] = [
  {
    id: 'arcee-ai/trinity-large-preview:free',
    name: 'Arcee AI: Trinity Large Preview (free)',
    context_length: 131000,
  },
  {
    id: 'arcee-ai/trinity-mini:free',
    name: 'Arcee AI: Trinity Mini (free)',
    context_length: 131072,
  },
  {
    id: 'nvidia/nemotron-3-nano-30b-a3b:free',
    name: 'NVIDIA: Nemotron 3 Nano 30B A3B (free)',
    context_length: 256000,
  },
  {
    id: 'liquid/lfm-2.5-1.2b-thinking:free',
    name: 'LiquidAI: LFM2.5-1.2B-Thinking (free)',
    context_length: 32768,
  },
  {
    id: 'liquid/lfm-2.5-1.2b-instruct:free',
    name: 'LiquidAI: LFM2.5-1.2B-Instruct (free)',
    context_length: 32768,
  },
  {
    id: 'qwen/qwen3-4b:free',
    name: 'Qwen: Qwen3 4B (free)',
    context_length: 40960,
  },
]

export interface ServiceStatus {
  status: 'ok' | 'error'
  latency_ms: number
  detail?: string
}

export interface ServiceHealth {
  qdrant: ServiceStatus
  langfuse: ServiceStatus
  openrouter: ServiceStatus
  postgres: ServiceStatus
}

export interface IngestionStatus {
  total_documents: number
  indexed_documents: number
  indexed_chunks: number
  status: string
  detail?: string
}

interface AppState {
  messages: Message[]
  isLoading: boolean
  health: ServiceHealth | null
  ingestion: IngestionStatus | null
  freeModels: FreeModelOption[]
  company: string
  year: string
  model: string
  addMessage: (msg: Message) => void
  setLoading: (v: boolean) => void
  setHealth: (h: ServiceHealth) => void
  setIngestion: (i: IngestionStatus) => void
  setFreeModels: (models: FreeModelOption[]) => void
  setCompany: (c: string) => void
  setYear: (y: string) => void
  setModel: (m: string) => void
  clearMessages: () => void
}

export const useAppStore = create<AppState>((set) => ({
  messages: [],
  isLoading: false,
  health: null,
  ingestion: null,
  freeModels: DEFAULT_FREE_MODELS,
  company: '',
  year: '',
  model: 'qwen/qwen3-4b:free',
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setLoading: (v) => set({ isLoading: v }),
  setHealth: (h) => set({ health: h }),
  setIngestion: (i) => set({ ingestion: i }),
  setFreeModels: (models) => set((state) => ({
    freeModels: models,
    model: models.some((option) => option.id === state.model) ? state.model : (models[0]?.id ?? state.model),
  })),
  setCompany: (c) => set({ company: c }),
  setYear: (y) => set({ year: y }),
  setModel: (m) => set({ model: m }),
  clearMessages: () => set({ messages: [] }),
}))
