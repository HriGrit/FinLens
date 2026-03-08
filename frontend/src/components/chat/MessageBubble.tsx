import { motion } from 'framer-motion'
import { CitationCard } from './CitationCard'
import { TokenAnalysis } from './TokenAnalysis'
import type { Message } from '../../stores/useAppStore'

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

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
        {message.content}
      </div>

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
    </motion.div>
  )
}
