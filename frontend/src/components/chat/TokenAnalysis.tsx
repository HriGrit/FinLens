import { Cpu, Clock, DollarSign, Zap } from 'lucide-react'
import type { TokenUsage } from '../../stores/useAppStore'

interface Props {
  usage: TokenUsage
  latency_ms?: number
  model?: string
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-muted">{icon}</span>
      <span className="font-mono text-[10px] text-text-secondary">{label}</span>
      <span className="font-mono text-[10px] text-amber-400">{value}</span>
    </div>
  )
}

export function TokenAnalysis({ usage, latency_ms, model }: Props) {
  const cost = usage.cost_usd != null ? `$${usage.cost_usd.toFixed(6)}` : '—'
  const latency = latency_ms != null ? `${(latency_ms / 1000).toFixed(2)}s` : '—'
  const provider =
    model?.startsWith('groq/') ? 'groq' : model?.startsWith('openrouter/') ? 'openrouter' : null
  const normalizedModel =
    provider === 'groq'
      ? model?.replace(/^groq\//, '')
      : provider === 'openrouter'
        ? model?.replace(/^openrouter\//, '')
        : model

  return (
    <div className="rounded-lg border border-border bg-bg p-3 mt-2">
      <div className="font-mono text-[9px] uppercase tracking-widest text-muted mb-2">Token Analysis</div>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        <Stat icon={<Cpu size={10} />} label="prompt" value={usage.prompt_tokens.toLocaleString()} />
        <Stat icon={<Cpu size={10} />} label="completion" value={usage.completion_tokens.toLocaleString()} />
        <Stat icon={<DollarSign size={10} />} label="cost" value={cost} />
        <Stat icon={<Clock size={10} />} label="latency" value={latency} />
        {provider && <Stat icon={<Zap size={10} />} label="provider" value={provider} />}
        {normalizedModel && <Stat icon={<Zap size={10} />} label="model" value={normalizedModel} />}
      </div>
    </div>
  )
}
