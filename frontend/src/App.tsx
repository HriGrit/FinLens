import { Sidebar } from './components/layout/Sidebar'
import { ChatPanel } from './components/chat/ChatPanel'

export default function App() {
  return (
    <div
      className="flex flex-col h-screen"
      style={{ backgroundColor: '#0d0f12' }}
    >
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#1e2330] flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm font-semibold text-[#f59e0b]">FinLens</span>
          <span className="font-mono text-[10px] text-[#6b7280] uppercase tracking-widest">
            Financial Document RAG
          </span>
        </div>
        <span className="font-mono text-[10px] text-[#6b7280]">v0.1.0</span>
      </header>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex flex-1 min-w-0 min-h-0">
          <ChatPanel />
        </main>
      </div>
    </div>
  )
}
