import { useState } from 'react'
import { MessageSquare, BookOpen, GitBranch, X, Github, ExternalLink } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { Sidebar } from './components/layout/Sidebar'
import { ChatPanel } from './components/chat/ChatPanel'
import { SecTenKExplainer } from './components/docs/SecTenKExplainer'
import { PipelineDoc } from './components/docs/PipelineDoc'

type AppView = 'chat' | 'sec-explainer' | 'pipeline'

const NAV_TABS = [
  { id: 'chat' as AppView, label: 'Chat', Icon: MessageSquare },
  { id: 'sec-explainer' as AppView, label: '10-K Guide', Icon: BookOpen },
  { id: 'pipeline' as AppView, label: 'Pipeline', Icon: GitBranch },
]

interface NavNudgeProps {
  view: AppView
  onNavigate: (id: AppView) => void
  onDismiss: () => void
}

function NavNudge({ view, onNavigate, onDismiss }: NavNudgeProps) {
  const otherTabs = NAV_TABS.filter(t => t.id !== view)
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.2 }}
      className="absolute top-full right-0 mt-2 z-50 bg-[#0d0f12] border border-amber-500/30 rounded-xl shadow-lg shadow-amber-500/10 p-4 w-64"
    >
      {/* Arrow */}
      <div className="absolute -top-1.5 right-6 w-3 h-3 bg-[#0d0f12] border-l border-t border-amber-500/30 rotate-45" />

      {/* Dismiss button */}
      <button
        onClick={onDismiss}
        className="absolute top-2 right-2 text-[#6b7280] hover:text-[#9ca3af] transition-colors"
        aria-label="Dismiss"
      >
        <X size={13} />
      </button>

      <p className="font-mono text-xs uppercase tracking-widest text-amber-400 mb-1">
        Explore FinLens
      </p>
      <p className="font-sans text-sm text-[#9ca3af] mb-3 leading-relaxed">
        There's more to explore →
      </p>

      <div className="flex flex-col gap-1.5">
        {otherTabs.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-md font-mono text-xs uppercase tracking-widest text-[#9ca3af] hover:text-amber-400 border border-[#1e2330] hover:border-amber-500/30 transition-colors"
          >
            <Icon size={11} />
            {label}
          </button>
        ))}
      </div>
    </motion.div>
  )
}

export default function App() {
  const [view, setView] = useState<AppView>('chat')
  const [nudgeDismissed, setNudgeDismissed] = useState(
    () => localStorage.getItem('finlens_nav_hint_seen') === '1'
  )

  const dismissNudge = () => {
    localStorage.setItem('finlens_nav_hint_seen', '1')
    setNudgeDismissed(true)
  }

  const navigateTo = (id: AppView) => {
    setView(id)
    if (!nudgeDismissed) dismissNudge()
  }

  return (
    <div
      className="flex flex-col h-screen"
      style={{ backgroundColor: '#0d0f12' }}
    >
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#1e2330] flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="font-mono text-base font-semibold text-[#f59e0b]">FinLens</span>
          <span className="font-mono text-xs text-[#6b7280] uppercase tracking-widest">
            Financial Document RAG
          </span>
        </div>
        <nav className="flex items-center gap-1 relative">
          {NAV_TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => navigateTo(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-mono text-xs uppercase tracking-widest transition-colors ${
                view === id
                  ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                  : 'text-[#6b7280] hover:text-[#9ca3af] border border-transparent'
              }`}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
          <AnimatePresence>
            {!nudgeDismissed && (
              <NavNudge view={view} onNavigate={navigateTo} onDismiss={dismissNudge} />
            )}
          </AnimatePresence>
        </nav>
      </header>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex flex-1 min-w-0 min-h-0 overflow-hidden">
          {view === 'chat' && <ChatPanel />}
          {view === 'sec-explainer' && <SecTenKExplainer />}
          {view === 'pipeline' && <PipelineDoc />}
        </main>
      </div>

      {/* Footer */}
      <footer className="flex items-center justify-between px-6 py-2 border-t border-[#1e2330] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500/60" />
          <span className="font-mono text-xs text-[#4b5563]">Built by</span>
          <a
            href="https://ayush-portfolio-mu-ten.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 font-mono text-xs text-amber-400/80 hover:text-amber-400 transition-colors"
          >
            Ayush Soam
            <ExternalLink size={10} />
          </a>
        </div>
        <a
          href="https://github.com/HriGrit/FinLens"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 font-mono text-xs text-[#4b5563] hover:text-[#9ca3af] transition-colors"
        >
          <Github size={12} />
          <span>Source</span>
        </a>
      </footer>
    </div>
  )
}
