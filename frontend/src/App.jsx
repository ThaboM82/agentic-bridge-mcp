import React from 'react';
import WorkflowCanvas from './components/WorkflowCanvas';
import { Cpu } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Enterprise Header Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-600/10 border border-indigo-500/30 rounded-xl shadow-inner">
            <Cpu className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              AgenticBridge-MCP
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                v1.0.0
              </span>
            </h1>
            <p className="text-xs text-slate-400">Enterprise Autonomous Agent Orchestrator & Multi-File Code Generation</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
            Orchestrator Online
          </span>
        </div>
      </header>

      {/* Main Dashboard Canvas Container */}
      <main className="flex-1 py-8">
        <WorkflowCanvas />
      </main>

      {/* Enterprise Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500 bg-slate-900/30">
        AgenticBridge-MCP &copy; 2026 Enterprise Architecture &bull; Powered by FastAPI, LangGraph, PostgreSQL pgvector, and React
      </footer>
    </div>
  );
}import React from 'react';
import WorkflowCanvas from './components/WorkflowCanvas';
import { Cpu } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Enterprise Header Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-600/10 border border-indigo-500/30 rounded-xl shadow-inner">
            <Cpu className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              AgenticBridge-MCP
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                v1.0.0
              </span>
            </h1>
            <p className="text-xs text-slate-400">Enterprise Autonomous Agent Orchestrator & Multi-File Code Generation</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
            Orchestrator Online
          </span>
        </div>
      </header>

      {/* Main Dashboard Canvas Container */}
      <main className="flex-1 py-8">
        <WorkflowCanvas />
      </main>

      {/* Enterprise Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500 bg-slate-900/30">
        AgenticBridge-MCP &copy; 2026 Enterprise Architecture &bull; Powered by FastAPI, LangGraph, PostgreSQL pgvector, and React
      </footer>
    </div>
  );
}