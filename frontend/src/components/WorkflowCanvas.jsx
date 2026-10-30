import React, { useState } from 'react';
import { Play, Loader2, CheckCircle2, AlertTriangle, FileCode, Terminal, Cpu } from 'lucide-react';

export default function WorkflowCanvas() {
  const [briefId, setBriefId] = useState('brief-' + Math.random().toString(36).substring(2, 9));
  const [rawBriefText, setRawBriefText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleExecute = async (e) => {
    e.preventDefault();
    if (!rawBriefText.trim()) {
      setError('Please provide a valid product brief before executing the workflow.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedFile(null);

    try {
      const response = await fetch('http://localhost:8000/workflow/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          brief_id: briefId,
          raw_brief_text: rawBriefText,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Workflow execution failed on the server.');
      }

      const data = await response.json();
      setResult(data);
      
      const fileKeys = Object.keys(data.generated_code_files || {});
      if (fileKeys.length > 0) {
        setSelectedFile(fileKeys[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Workflow Orchestration Canvas
          </h2>
          {loading && (
            <span className="badge badge-sky animate-pulse-glow flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping"></span>
              Pipeline Running...
            </span>
          )}
        </div>
        <p className="text-slate-400 text-sm mb-6">
          Submit an enterprise product brief to trigger autonomous ingestion, vector design analysis, multi-file code generation, and publishing.
        </p>

        <form onSubmit={handleExecute} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Brief ID</label>
            <input
              type="text"
              value={briefId}
              onChange={(e) => setBriefId(e.target.value)}
              className="input-enterprise font-mono"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Product Brief / User Story Text</label>
            <textarea
              rows={5}
              value={rawBriefText}
              onChange={(e) => setRawBriefText(e.target.value)}
              placeholder="Describe your enterprise application requirements, architecture specs, and feature sets..."
              className="input-enterprise"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full py-3 text-base shadow-indigo-600/30"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Executing Agentic Pipeline...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Execute Workflow Pipeline
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-rose-950/40 border border-rose-800/50 rounded-xl flex items-center gap-3 text-rose-300 text-sm shadow-lg">
            <AlertTriangle className="w-5 h-5 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
          {/* Left Column: Requirements & Execution Logs */}
          <div className="space-y-6">
            {/* Requirements Card */}
            <div className="glass-panel p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  Extracted Requirements
                </h3>
                <span className="badge badge-emerald">
                  {result.extracted_requirements?.length || 0} Parsed
                </span>
              </div>
              <ul className="space-y-2.5">
                {result.extracted_requirements?.map((req, idx) => (
                  <li key={idx} className="text-sm text-slate-300 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 flex items-start gap-2.5">
                    <span className="text-indigo-400 font-mono text-xs mt-0.5">#{idx + 1}</span>
                    <span className="flex-1">{req}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Execution Logs */}
            <div className="glass-panel p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Terminal className="w-5 h-5 text-indigo-400" />
                  Execution Logs
                </h3>
                <span className="badge badge-indigo">
                  {result.review_status || 'completed'}
                </span>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-300 space-y-1.5 max-h-60 overflow-y-auto border border-slate-800 shadow-inner">
                {result.execution_logs?.map((log, idx) => (
                  <div key={idx} className={log.includes('ERROR') ? 'text-rose-400' : log.includes('SUCCESS') ? 'text-emerald-400' : 'text-slate-400'}>
                    {log}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Generated Code Viewer */}
          <div className="glass-panel p-6 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <FileCode className="w-5 h-5 text-sky-400" />
                Generated Code Artifacts
              </h3>
              <span className="badge badge-sky">
                {Object.keys(result.generated_code_files || {}).length} Files
              </span>
            </div>

            {/* File Tabs */}
            <div className="flex flex-wrap gap-2 mb-4">
              {Object.keys(result.generated_code_files || {}).map((filename) => (
                <button
                  key={filename}
                  onClick={() => setSelectedFile(filename)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                    selectedFile === filename
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 border border-indigo-500'
                      : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-white border border-slate-800'
                  }`}
                >
                  {filename}
                </button>
              ))}
            </div>

            {/* File Content Code Block */}
            <div className="flex-1 bg-slate-950 rounded-xl p-4 font-mono text-xs text-slate-300 overflow-x-auto max-h-[480px] border border-slate-800 shadow-inner">
              <pre>
                {selectedFile && result.generated_code_files[selectedFile]
                  ? result.generated_code_files[selectedFile]
                  : 'Select a generated file above to inspect source code.'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}