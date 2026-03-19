import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { ServiceGrid } from '../health/ServiceGrid'
import { IngestionMeter } from '../ingestion/IngestionMeter'
import { DEFAULT_PROVIDER_MODELS, type LlmProvider, useAppStore } from '../../stores/useAppStore'
import { usePolling } from '../../hooks/usePolling'
import { getModels, getServicesStatus, getIngestionStatus } from '../../api/client'

export function Sidebar() {
  const {
    health,
    ingestion,
    freeModels,
    company,
    year,
    model,
    provider,
    setHealth,
    setIngestion,
    setFreeModels,
    setCompany,
    setYear,
    setModel,
    setProvider,
  } = useAppStore()
  const [isLoadingModels, setIsLoadingModels] = useState(true)
  const [modelLoadError, setModelLoadError] = useState<string | null>(null)

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

    const loadModels = async () => {
      try {
        const models = await getModels(provider)
        if (!cancelled) {
          setFreeModels(models.length ? models : DEFAULT_PROVIDER_MODELS[provider])
          setModelLoadError(null)
        }
      } catch (error) {
        if (!cancelled) {
          setFreeModels(DEFAULT_PROVIDER_MODELS[provider])
          const msg = error instanceof Error ? error.message : 'Unknown error while loading model list.'
          setModelLoadError(`Using fallback ${provider} models. (${msg})`)
        }
      } finally {
        if (!cancelled) {
          setIsLoadingModels(false)
        }
      }
    }

    void loadModels()

    return () => {
      cancelled = true
    }
  }, [provider, setFreeModels, setModel])

  const providerLabel = provider === 'groq' ? 'Groq' : 'OpenRouter'

  const onChangeProvider = (nextProvider: LlmProvider) => {
    setProvider(nextProvider)
    setModel(DEFAULT_PROVIDER_MODELS[nextProvider][0]?.id ?? '')
    setIsLoadingModels(true)
  }

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
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-amber-300">Active Provider</div>
          <div className="mt-1 flex items-center justify-between">
            <span className="font-mono text-sm text-text-primary">{providerLabel}</span>
            <span className="rounded-full border border-amber-500/30 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-amber-300">
              {freeModels.length} models
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted">Provider</label>
          <select
            value={provider}
            onChange={(e) => onChangeProvider(e.target.value as LlmProvider)}
            className="w-full bg-bg border border-border rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-amber-500/50"
          >
            <option value="openrouter">OpenRouter</option>
            <option value="groq">Groq</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full bg-bg border border-border rounded px-2 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-amber-500/50"
          >
            {isLoadingModels && (
              <option value={model} disabled>
                Loading {providerLabel} models...
              </option>
            )}
            {!isLoadingModels &&
              freeModels.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
          </select>
          {modelLoadError && (
            <p className="mt-1 text-[10px] text-amber-400">Could not load live model list. Using fallback {providerLabel} models.</p>
          )}
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
