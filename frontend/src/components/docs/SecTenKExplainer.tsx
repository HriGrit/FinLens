import { motion } from 'framer-motion'
import {
  FileText,
  BarChart2,
  Users,
  BookOpen,
  DollarSign,
  TrendingUp,
  ChevronRight,
  Archive,
  Info,
} from 'lucide-react'

const PARTS = [
  {
    part: 'Part I',
    title: 'Business Overview',
    icon: 'Building2',
    sections: [
      'Business Description',
      'Risk Factors',
      'Unresolved Staff Comments',
      'Properties',
      'Legal Proceedings',
    ],
  },
  {
    part: 'Part II',
    title: 'Financial Performance',
    icon: 'BarChart2',
    sections: [
      'MD&A',
      'Market Risk Disclosures',
      'Financial Statements & Notes',
      'Controls & Procedures',
    ],
  },
  {
    part: 'Part III',
    title: 'Governance & Compensation',
    icon: 'Users',
    sections: [
      'Directors & Executives',
      'Executive Compensation',
      'Security Ownership',
      'Related Transactions',
    ],
  },
  {
    part: 'Part IV',
    title: 'Exhibits & Schedules',
    icon: 'Archive',
    sections: ['Exhibits List', 'Financial Statement Schedules'],
  },
]

const KEY_METRICS = [
  'Revenue',
  'EPS',
  'Total Assets',
  'Operating Cash Flow',
  'Net Income',
  'CapEx',
  'Operating Margin',
  'Debt/Equity',
]

const WHY_IT_MATTERS = [
  'Required by SEC for all public companies with $10M+ in assets and 2,000+ shareholders',
  'Provides audited financials — the gold standard for investment decisions',
  'Risk Factors section discloses material threats to the business',
  "MD&A gives management's own interpretation of results and outlook",
  'Footnotes reveal accounting policies, off-balance-sheet items, and contingencies',
]

const PART_ICONS: Record<string, React.ReactNode> = {
  Building2: <FileText size={16} className="text-amber-400" />,
  BarChart2: <BarChart2 size={16} className="text-amber-400" />,
  Users: <Users size={16} className="text-amber-400" />,
  Archive: <Archive size={16} className="text-amber-400" />,
}

function PartCard({ part, index }: { part: (typeof PARTS)[number]; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
      className="rounded-xl border border-border bg-surface p-5 flex flex-col gap-3"
    >
      <div className="flex items-center gap-2">
        {PART_ICONS[part.icon]}
        <span className="font-mono text-xs text-amber-400 uppercase tracking-widest">{part.part}</span>
      </div>
      <p className="font-semibold text-text-primary text-base">{part.title}</p>
      <ul className="space-y-1">
        {part.sections.map((s) => (
          <li key={s} className="flex items-center gap-1.5 text-sm text-text-secondary font-sans">
            <ChevronRight size={10} className="text-muted flex-shrink-0" />
            {s}
          </li>
        ))}
      </ul>
    </motion.div>
  )
}

export function SecTenKExplainer() {
  return (
    <div className="flex-1 overflow-y-auto px-8 py-8 w-full">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-3">
          <FileText size={22} className="text-amber-400" />
          <h1 className="font-mono text-2xl font-bold text-text-primary">SEC Form 10-K</h1>
        </div>
        <p className="text-base text-text-secondary font-sans leading-relaxed max-w-2xl">
          The 10-K is a comprehensive annual report that public companies must file with the SEC within 60–90 days of
          their fiscal year end. It is the most detailed picture of a company's business, risk profile, and financial
          health available to investors.
        </p>
      </motion.div>

      {/* Parts grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        {PARTS.map((p, i) => (
          <PartCard key={p.part} part={p} index={i} />
        ))}
      </div>

      {/* Key metrics */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.35 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 mb-3">
          <DollarSign size={14} className="text-amber-400" />
          <p className="font-mono text-xs uppercase tracking-widest text-muted">Key Metrics in a 10-K</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {KEY_METRICS.map((m) => (
            <span
              key={m}
              className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-sm font-mono text-amber-300"
            >
              {m}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Why it matters */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.42 }}
        className="rounded-xl border border-border bg-surface p-5"
      >
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={14} className="text-amber-400" />
          <p className="font-mono text-xs uppercase tracking-widest text-muted">Why It Matters</p>
        </div>
        <ul className="space-y-3">
          {WHY_IT_MATTERS.map((item) => (
            <li key={item} className="flex items-start gap-2 text-base text-text-secondary font-sans">
              <Info size={13} className="text-amber-400 flex-shrink-0 mt-0.5" />
              {item}
            </li>
          ))}
        </ul>
        <div className="mt-5 pt-4 border-t border-border flex items-center gap-2 text-sm text-muted font-mono">
          <BookOpen size={11} />
          <span>FinLens is trained on 10-K filings from FinanceBench — ask questions about any indexed company.</span>
        </div>
      </motion.div>
    </div>
  )
}
