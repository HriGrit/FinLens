import type { IngestionStatus } from '../../stores/useAppStore'

export function IngestionMeter({ ingestion }: { ingestion: IngestionStatus | null }) {
  const total = ingestion?.total_documents ?? 0
  const indexed = ingestion?.indexed_documents ?? 0
  const chunks = ingestion?.indexed_chunks ?? 0
  const pct = total > 0 ? Math.round((indexed / total) * 100) : 0

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-secondary">Ingestion</span>
        <span className="font-mono text-[10px] text-amber-500">{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-border overflow-hidden">
        <div
          className="h-full rounded-full bg-amber-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex items-center justify-between font-mono text-[10px] text-text-secondary">
        <span>
          {indexed} / {total} docs
        </span>
        <span>{chunks.toLocaleString()} chunks</span>
      </div>
    </div>
  )
}
