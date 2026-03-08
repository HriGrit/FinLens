import { Database, Activity, Key, Server } from 'lucide-react'
import { clsx } from 'clsx'
import type { ServiceHealth, ServiceStatus } from '../../stores/useAppStore'

interface CardProps {
  label: string
  icon: React.ReactNode
  status: ServiceStatus | undefined
}

function ServiceCard({ label, icon, status }: CardProps) {
  const s = status?.status ?? 'loading'
  return (
    <div
      className={clsx(
        'flex flex-col gap-1 rounded-lg border p-3 bg-surface',
        s === 'ok' && 'border-emerald-500/20',
        s === 'error' && 'border-red-500/20',
        s === 'loading' && 'border-border',
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-text-secondary font-mono text-[10px] uppercase tracking-widest">{label}</span>
        <span
          className={clsx(
            'w-2 h-2 rounded-full',
            s === 'ok' && 'bg-emerald-400',
            s === 'error' && 'bg-red-400',
            s === 'loading' && 'bg-gray-500 animate-pulse',
          )}
        />
      </div>
      <div className="flex items-center gap-2 text-text-primary">
        <span className="text-text-secondary">{icon}</span>
        <span className="font-mono text-xs">
          {s === 'loading' ? '—' : s === 'ok' ? 'online' : 'error'}
        </span>
      </div>
      {status?.latency_ms !== undefined && status.latency_ms >= 0 && (
        <span className="font-mono text-[10px] text-muted">{status.latency_ms}ms</span>
      )}
    </div>
  )
}

export function ServiceGrid({ health }: { health: ServiceHealth | null }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <ServiceCard label="Qdrant" icon={<Database size={12} />} status={health?.qdrant} />
      <ServiceCard label="Langfuse" icon={<Activity size={12} />} status={health?.langfuse} />
      <ServiceCard label="OpenRouter" icon={<Key size={12} />} status={health?.openrouter} />
      <ServiceCard label="Postgres" icon={<Server size={12} />} status={health?.postgres} />
    </div>
  )
}
