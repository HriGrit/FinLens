import { create } from 'zustand'

export interface Citation {
  index: number
  company: string
  year: string
  doc_type: string
  page_number: number
  filename: string
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
  citations?: Citation[]
  usage?: TokenUsage
  model?: string
  latency_ms?: number
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  usage: TokenUsage
  model: string
  latency_ms: number
}

export interface FreeModelOption {
  id: string
  name: string
  context_length?: number
  description?: string
}

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
  freeModels: [],
  company: '',
  year: '',
  model: 'openrouter/free',
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
