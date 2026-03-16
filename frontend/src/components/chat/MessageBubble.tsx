import { motion } from 'framer-motion'
import { CitationCard } from './CitationCard'
import { TokenAnalysis } from './TokenAnalysis'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import type { Message } from '../../stores/useAppStore'

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const reasoning = message.reasoning

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex flex-col gap-3 ${isUser ? 'items-end' : 'items-start'}`}
    >
      {/* Role label */}
      <div className="font-mono text-[9px] uppercase tracking-widest text-muted px-1">
        {isUser ? 'you' : 'finlens'}
      </div>

      {/* Message body */}
      <div
        className={`rounded-xl px-4 py-3 max-w-[85%] text-sm leading-relaxed font-sans ${
          isUser
            ? 'bg-amber-500/10 border border-amber-500/20 text-text-primary ml-8'
            : 'bg-surface border border-border text-text-primary mr-8'
        }`}
      >
        {isUser ? (
          message.content
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[[rehypeSanitize]]}
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc pl-5 mb-2 last:mb-0">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 last:mb-0">{children}</ol>,
              li: ({ children }) => <li className="mb-1 last:mb-0">{children}</li>,
              a: ({ href, children }) => (
                <a href={href ?? '#'} className="text-amber-300 hover:text-amber-200 underline">
                  {children}
                </a>
              ),
              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
              em: ({ children }) => <em className="italic">{children}</em>,
              code: ({ children }) => <code className="rounded bg-bg px-1 py-0.5">{children}</code>,
              hr: () => <hr className="my-4 border-border" />,
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
      </div>

      {!isUser && message.fallback && message.fallback.events.length > 0 && (
        <div className="w-full max-w-[85%] text-xs">
          {message.fallback.events.map((event, idx) => (
            <p key={`${event.from_model}-${idx}`} className="font-mono text-amber-300">
              {event.from_model} is very hot, moving to {event.to_model}.
            </p>
          ))}
        </div>
      )}

      {/* Citations */}
      {!isUser && message.citations && message.citations.length > 0 && (
        <div className="flex flex-col gap-2 w-full max-w-[85%]">
          {message.citations.map((c) => (
            <CitationCard key={c.index} citation={c} />
          ))}
        </div>
      )}

      {/* Token analysis */}
      {!isUser && message.usage && (
        <div className="w-full max-w-[85%]">
          <TokenAnalysis usage={message.usage} latency_ms={message.latency_ms} model={message.model} />
        </div>
      )}

      {!isUser && reasoning && (
        <details className="w-full max-w-[85%] bg-surface border border-border rounded-lg px-3 py-2 text-xs">
          <summary className="cursor-pointer font-mono text-muted uppercase tracking-widest">Reasoning</summary>
          <div className="mt-3 space-y-3 text-text-secondary font-sans">
            <div>
              <p className="font-mono text-muted mb-1">Summary</p>
              <p>{reasoning.summary}</p>
              {message.trace_id && <p className="mt-1 text-[11px]">Trace: {message.trace_id}</p>}
            </div>
            <div className="grid gap-2 text-[11px]">
              <p>
                Retrieval: top_k={reasoning.retrieval.retrieval_top_k}, candidates returned={reasoning.retrieval.retrieved_candidates}
              </p>
              <p>
                Rerank: requested={reasoning.rerank.requested_rerank_top_k}, selected={reasoning.rerank.selected_nodes}
              </p>
              <p>
                Model: {reasoning.generation.model}, generation latency {reasoning.generation.latency_ms}ms
              </p>
            </div>
            {reasoning.rerank.top_sources.length > 0 && (
              <div>
                <p className="font-mono text-muted mb-1">Top Sources</p>
                <div className="space-y-1">
                  {reasoning.rerank.top_sources.map((source, idx) => {
                    const hasLocation = source.filename && source.page_number
                    return (
                      <div key={`${source.filename ?? 'source'}-${idx}`} className="font-mono">
                        • {source.filename || 'Unknown document'}
                        {hasLocation ? ` (p.${source.page_number})` : ''}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </details>
      )}
    </motion.div>
  )
}
