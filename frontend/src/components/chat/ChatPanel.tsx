import axios from 'axios'
import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2, Trash2 } from 'lucide-react'
import { MessageBubble } from './MessageBubble'
import { useAppStore } from '../../stores/useAppStore'
import { postChat } from '../../api/client'
import type { AxiosError } from 'axios'

function SkeletonMessage() {
  return (
    <div className="flex flex-col gap-2 items-start">
      <div className="font-mono text-[9px] uppercase tracking-widest text-muted px-1">finlens</div>
      <div className="flex flex-col gap-2 w-64">
        <div className="h-3 bg-surface rounded animate-pulse" />
        <div className="h-3 bg-surface rounded animate-pulse w-5/6" />
        <div className="h-3 bg-surface rounded animate-pulse w-4/6" />
      </div>
    </div>
  )
}

export function ChatPanel() {
  const [input, setInput] = useState('')
  const { messages, isLoading, addMessage, setLoading, clearMessages, company, year, model } = useAppStore()
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const pendingRequest = useRef<AbortController | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    return () => {
      pendingRequest.current?.abort()
      pendingRequest.current = null
    }
  }, [])

  const submit = async () => {
    const query = input.trim()
    if (!query || isLoading) return

    setInput('')
    addMessage({ id: crypto.randomUUID(), role: 'user', content: query })
    setLoading(true)
    const controller = new AbortController()
    pendingRequest.current = controller

    try {
      const result = await postChat(
        {
        query,
        company: company || undefined,
        year: year || undefined,
        model: model || undefined,
        },
        controller.signal,
      )
      addMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.answer,
        citations: result.citations,
        reasoning: result.reasoning,
        trace_id: result.trace_id,
        usage: result.usage,
        model: result.model,
        latency_ms: result.latency_ms,
      })
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const axiosError = err as AxiosError<{ detail?: string } | string>
        if (axiosError.code === 'ERR_CANCELED') return

        const responseData = axiosError.response?.data
        const msg =
          (typeof responseData === 'object' && responseData !== null && 'detail' in responseData && typeof responseData.detail === 'string'
            ? responseData.detail
            : typeof responseData === 'string'
              ? responseData
              : axiosError.message) || 'Request failed'

        addMessage({
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Error: ${msg}`,
        })
        return
      }

      const msg = err instanceof Error ? err.message : 'Request failed'
      addMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${msg}`,
      })
    } finally {
      if (pendingRequest.current === controller) {
        pendingRequest.current = null
      }
      setLoading(false)
    }
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border">
        <span className="font-mono text-xs text-text-secondary uppercase tracking-widest">Chat</span>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="flex items-center gap-1.5 text-muted hover:text-text-secondary transition-colors text-xs font-mono"
          >
            <Trash2 size={12} />
            clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-6">
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center h-full gap-4 text-center"
          >
            <div className="text-4xl font-mono font-semibold text-amber-500/20">FinLens</div>
            <p className="text-text-secondary text-sm font-sans max-w-sm">
              Ask questions about financial filings. Answers are grounded in SEC documents with citations.
            </p>
            <div className="flex flex-col gap-2 text-xs font-mono text-muted">
              <div className="border border-border rounded px-3 py-1.5 hover:border-amber-500/30 cursor-default">
                What was 3M's revenue in 2019?
              </div>
              <div className="border border-border rounded px-3 py-1.5 hover:border-amber-500/30 cursor-default">
                Compare operating margins across years
              </div>
            </div>
          </motion.div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && <SkeletonMessage />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-border">
        <div className="flex items-end gap-3 rounded-xl border border-border bg-surface px-4 py-3 focus-within:border-amber-500/40 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about financial filings..."
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-text-primary placeholder:text-muted font-sans focus:outline-none"
            style={{ maxHeight: '120px', overflowY: 'auto' }}
          />
          <button
            onClick={submit}
            disabled={!input.trim() || isLoading}
            className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-lg bg-amber-500 text-bg disabled:opacity-30 disabled:cursor-not-allowed hover:bg-amber-400 transition-colors"
          >
            {isLoading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </div>
        <p className="mt-1.5 font-mono text-[9px] text-muted text-right">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}
