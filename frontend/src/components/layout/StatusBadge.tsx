import { clsx } from 'clsx'

interface Props {
  status: 'ok' | 'error' | 'loading'
  label?: string
}

export function StatusBadge({ status, label }: Props) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono font-medium',
        status === 'ok' && 'bg-emerald-500/10 text-emerald-400',
        status === 'error' && 'bg-red-500/10 text-red-400',
        status === 'loading' && 'bg-gray-500/10 text-gray-400',
      )}
    >
      <span
        className={clsx(
          'w-1.5 h-1.5 rounded-full',
          status === 'ok' && 'bg-emerald-400',
          status === 'error' && 'bg-red-400',
          status === 'loading' && 'bg-gray-400 animate-pulse',
        )}
      />
      {label ?? status}
    </span>
  )
}
