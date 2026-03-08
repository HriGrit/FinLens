import { useEffect } from 'react'
import { RefreshCw } from 'lucide-react'
import { ServiceGrid } from '../health/ServiceGrid'
import { IngestionMeter } from '../ingestion/IngestionMeter'
import { useAppStore } from '../../stores/useAppStore'
import { usePolling } from '../../hooks/usePolling'
import { getFreeModels, getServicesStatus, getIngestionStatus } from '../../api/client'

export function Sidebar() {
  const {
    health,
    ingestion,
    freeModels,
    company,
    year,
    model,
    setHealth,
    setIngestion,
    setFreeModels,
    setCompany,
    setYear,
    setModel,
  } = useAppStore()

  usePolling(async () => {
    try {
      const h = await getServicesStatus()
      setHealth(h)
    } catch { /* ignore */ }
    try {
      const i = await getIngestionStatus()
      setIngestion(i)
    } catch { /* ignore */ }
  }, 10_000)

  useEffect(() => {
    let cancelled = false

    void getFreeModels()
      .then((models) => {
        if (!cancelled) {
          setFreeModels(models)
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [setFreeModels])

  const COMPANIES = ['3M', 'Apple', 'Microsoft', 'Amazon', 'Google']
  const YEARS = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

  return (
    <aside className="w-56 flex-shrink-0 flex flex-col gap-5 p-4 border-r border-border bg-surface overflow-y-auto">
      {/* Services */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-text-secondary">Services</span>
          <RefreshCw size={10} className="text-muted" />
        </div>
        <ServiceGrid health={health} />
      </section>

      {/* Ingestion */}
      <section>
        <IngestionMeter ingestion={ingestion} />
      </section>

      {/* Filters */}
      <section className="flex flex-col gap-3">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-secondary">Filters</span>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full bg-bg border border-border rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-amber-500/50"
          >
            {freeModels.length === 0 && (
              <option value="openrouter/free">Loading free models...</option>
            )}
            {freeModels.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted">Company</label>
          <select
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            className="w-full bg-bg border border-border rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-amber-500/50"
          >
            <option value="">All companies</option>
            {COMPANIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted">Year</label>
          <select
            value={year}
            onChange={(e) => setYear(e.target.value)}
            className="w-full bg-bg border border-border rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-amber-500/50"
          >
            <option value="">All years</option>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
      </section>
    </aside>
  )
}
