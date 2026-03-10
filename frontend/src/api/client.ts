import axios from 'axios'
import type { ChatResponse, FreeModelOption, IngestionStatus, ServiceHealth } from '../stores/useAppStore'

const api = axios.create({
  baseURL: '/',
  timeout: 60_000,
})

export interface ChatRequest {
  query: string
  company?: string
  year?: string
  model?: string
  retrieval_top_k?: number
  rerank_top_k?: number
}

export async function postChat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', req, { signal })
  return data
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  const { data } = await api.get<IngestionStatus>('/status/ingestion')
  return data
}

export async function getServicesStatus(): Promise<ServiceHealth> {
  const { data } = await api.get<ServiceHealth>('/status/services')
  return data
}

export async function getFreeModels(): Promise<FreeModelOption[]> {
  const { data } = await api.get<FreeModelOption[]>('/models/free')
  return data
}
