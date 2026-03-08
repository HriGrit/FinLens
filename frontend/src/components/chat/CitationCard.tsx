import { FileText } from 'lucide-react'
import type { Citation } from '../../stores/useAppStore'

export function CitationCard({ citation }: { citation: Citation }) {
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-bg p-3">
      <div className="flex items-center gap-2">
        <FileText size={12} className="text-amber-500 flex-shrink-0" />
        <span className="font-mono text-[10px] text-amber-500">
          [{citation.index}]
        </span>
        <span className="font-mono text-[10px] text-text-secondary truncate">
          {citation.company} · {citation.year} · {citation.doc_type} · p.{citation.page_number}
        </span>
      </div>
      {citation.excerpt && (
        <p className="text-xs text-text-secondary font-sans leading-relaxed line-clamp-3 pl-5">
          "{citation.excerpt}"
        </p>
      )}
      <span className="pl-5 font-mono text-[9px] text-muted truncate">{citation.filename}</span>
    </div>
  )
}
