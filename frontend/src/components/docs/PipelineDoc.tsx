import { motion } from 'framer-motion'
import {
  FileText,
  Scissors,
  Layers,
  Database,
  Search,
  Sliders,
  Zap,
  MessageSquare,
} from 'lucide-react'

const PIPELINE_STEPS = [
  { id: 1, label: 'PDF Ingest', icon: 'FileText', color: 'amber', desc: 'Docling parses PDF → LlamaIndex TextNodes (6 required metadata fields).' },
  { id: 2, label: 'Chunk', icon: 'Scissors', color: 'amber', desc: 'SemanticSplitterNodeParser splits paragraph nodes; tables pass through.' },
  { id: 3, label: 'Embed', icon: 'Layers', color: 'emerald', desc: 'Alibaba-NLP/gte-modernbert-base (768-dim) cosine vectors.' },
  { id: 4, label: 'Index', icon: 'Database', color: 'emerald', desc: 'Qdrant dense index + BM25 sparse index pickled to disk.' },
  { id: 5, label: 'Hybrid Retrieve', icon: 'Search', color: 'amber', desc: 'BM25 + Qdrant dense retrieval fused via RRF (k=60).' },
  { id: 6, label: 'Rerank', icon: 'Sliders', color: 'amber', desc: 'cross-encoder/ms-marco-MiniLM-L-6-v2 scores top candidates.' },
  { id: 7, label: 'Generate', icon: 'Zap', color: 'emerald', desc: 'LiteLLM → OpenRouter → Qwen3/Mistral with citation assembly.' },
  { id: 8, label: 'Answer', icon: 'MessageSquare', color: 'emerald', desc: 'Grounded answer with citations + Langfuse trace ID.' },
]

const STEP_ICONS: Record<string, React.ReactNode> = {
  FileText: <FileText size={18} />,
  Scissors: <Scissors size={18} />,
  Layers: <Layers size={18} />,
  Database: <Database size={18} />,
  Search: <Search size={18} />,
  Sliders: <Sliders size={18} />,
  Zap: <Zap size={18} />,
  MessageSquare: <MessageSquare size={18} />,
}

const TECH_STACK = [
  { name: 'Docling', role: 'PDF Parsing', color: 'amber' },
  { name: 'LlamaIndex', role: 'Node Abstractions', color: 'amber' },
  { name: 'Qdrant', role: 'Vector Store', color: 'emerald' },
  { name: 'LiteLLM', role: 'LLM Abstraction', color: 'emerald' },
  { name: 'FastAPI', role: 'API Server', color: 'blue' },
  { name: 'Langfuse', role: 'Observability', color: 'purple' },
]

type Color = 'amber' | 'emerald' | 'blue' | 'purple'

const COLOR_CLASSES: Record<Color, { bg: string; border: string; text: string; icon: string }> = {
  amber: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-300',
    icon: 'text-amber-400',
  },
  emerald: {
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    text: 'text-emerald-300',
    icon: 'text-emerald-400',
  },
  blue: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
    text: 'text-blue-300',
    icon: 'text-blue-400',
  },
  purple: {
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/30',
    text: 'text-purple-300',
    icon: 'text-purple-400',
  },
}

function Badge({ label, color }: { label: string; color: Color }) {
  const c = COLOR_CLASSES[color]
  return (
    <span className={`rounded-full border ${c.border} ${c.bg} px-2.5 py-0.5 text-xs font-mono ${c.text}`}>
      {label}
    </span>
  )
}

function AnimatedArrow() {
  return (
    <div className="flex-shrink-0 flex items-center px-1">
      <motion.svg
        width="24"
        height="12"
        viewBox="0 0 24 12"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        initial="hidden"
        animate="visible"
      >
        <motion.path
          d="M0 6 H18 M14 2 L20 6 L14 10"
          stroke="#f59e0b"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          variants={{
            hidden: { pathLength: 0, opacity: 0 },
            visible: { pathLength: 1, opacity: 1 },
          }}
          transition={{ duration: 0.4, ease: 'easeInOut' }}
        />
      </motion.svg>
    </div>
  )
}

function StepCard({ step, index }: { step: (typeof PIPELINE_STEPS)[number]; index: number }) {
  const c = COLOR_CLASSES[step.color as Color]
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.07 }}
      className={`flex-shrink-0 w-32 rounded-xl border ${c.border} ${c.bg} p-3 flex flex-col items-center gap-2`}
    >
      <span className={`${c.icon}`}>{STEP_ICONS[step.icon]}</span>
      <span className="font-mono text-xs text-muted text-center">{step.id}</span>
      <span className="font-sans text-sm text-center text-text-primary font-medium leading-snug">{step.label}</span>
    </motion.div>
  )
}

function PipelineFlow() {
  return (
    <div className="overflow-x-auto pb-3">
      <div className="flex items-center min-w-max gap-0">
        {PIPELINE_STEPS.map((step, i) => (
          <div key={step.id} className="flex items-center">
            <StepCard step={step} index={i} />
            {i < PIPELINE_STEPS.length - 1 && <AnimatedArrow />}
          </div>
        ))}
      </div>
    </div>
  )
}

function StepDescriptions() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {PIPELINE_STEPS.map((step, i) => {
        const c = COLOR_CLASSES[step.color as Color]
        return (
          <motion.div
            key={step.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25, delay: 0.3 + i * 0.05 }}
            className="flex gap-3 items-start"
          >
            <span className={`flex-shrink-0 rounded-md border ${c.border} ${c.bg} p-1.5 ${c.icon}`}>
              {STEP_ICONS[step.icon]}
            </span>
            <div>
              <p className="text-sm font-semibold text-text-primary">{step.label}</p>
              <p className="text-sm text-text-secondary font-sans mt-0.5 leading-relaxed">{step.desc}</p>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

function ModelCard({
  name,
  dims,
  metric,
  role,
}: {
  name: string
  dims?: string
  metric?: string
  role: string
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4 flex flex-col gap-2">
      <p className="font-mono text-xs uppercase tracking-widest text-muted">{role}</p>
      <p className="font-semibold text-text-primary text-base">{name}</p>
      <div className="flex flex-wrap gap-1.5">
        {dims && <Badge label={dims} color="emerald" />}
        {metric && <Badge label={metric} color="amber" />}
      </div>
    </div>
  )
}

function RRFVisual() {
  const bars = [
    { label: 'BM25', pct: 60, color: 'bg-amber-500/60' },
    { label: 'Dense', pct: 75, color: 'bg-emerald-500/60' },
    { label: 'RRF', pct: 90, color: 'bg-amber-400' },
  ]
  return (
    <div className="rounded-xl border border-border bg-surface p-4 flex flex-col gap-3">
      <p className="font-mono text-xs uppercase tracking-widest text-muted">RRF Fusion (conceptual)</p>
      <div className="space-y-2">
        {bars.map((b) => (
          <div key={b.label} className="flex items-center gap-3">
            <span className="w-12 text-right font-mono text-xs text-muted">{b.label}</span>
            <div className="flex-1 h-2 rounded-full bg-surface border border-border overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${b.color}`}
                initial={{ width: 0 }}
                animate={{ width: `${b.pct}%` }}
                transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="text-sm text-text-secondary font-sans">
        Reciprocal Rank Fusion (k=60) merges sparse and dense rankings into a single ordered list.
      </p>
    </div>
  )
}

function TechStackGrid() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {TECH_STACK.map((t, i) => {
        const c = COLOR_CLASSES[t.color as Color]
        return (
          <motion.div
            key={t.name}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.25, delay: 0.4 + i * 0.06 }}
            className={`rounded-xl border ${c.border} ${c.bg} p-3 flex flex-col gap-1`}
          >
            <p className={`font-mono text-sm font-bold ${c.text}`}>{t.name}</p>
            <p className="font-sans text-sm text-text-secondary">{t.role}</p>
          </motion.div>
        )
      })}
    </div>
  )
}

export function PipelineDoc() {
  return (
    <div className="flex-1 overflow-y-auto px-8 py-8 w-full">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mb-8"
      >
        <h1 className="font-mono text-2xl font-bold text-text-primary mb-2">FinLens Pipeline</h1>
        <p className="text-base text-text-secondary font-sans leading-relaxed max-w-2xl">
          A hybrid RAG pipeline combining dense vector search, sparse BM25 retrieval, and cross-encoder reranking to
          answer questions over financial filings.
        </p>
      </motion.div>

      {/* Flow diagram */}
      <section className="mb-8">
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-4">Pipeline Flow</p>
        <PipelineFlow />
      </section>

      {/* Step descriptions */}
      <section className="mb-8">
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-4">Step Details</p>
        <StepDescriptions />
      </section>

      {/* Models + RRF */}
      <section className="mb-8">
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-4">Models</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <ModelCard
            name="gte-modernbert-base"
            dims="768-dim"
            metric="cosine"
            role="Embedding Model"
          />
          <ModelCard
            name="ms-marco-MiniLM-L-6-v2"
            role="Cross-Encoder Reranker"
          />
          <RRFVisual />
        </div>
      </section>

      {/* Tech stack */}
      <section>
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-4">Tech Stack</p>
        <TechStackGrid />
      </section>
    </div>
  )
}
