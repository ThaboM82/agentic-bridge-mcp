import React, { useState, useEffect } from 'react';
import { Search, PlusCircle, FileText, Database, Loader2 } from 'lucide-react';

interface DocumentResponse {
  id: string | number;
  title: string;
  content: string;
}

interface SearchResultItem extends DocumentResponse {
  similarity_score: number;
}

const apiClient = {
  async getDocuments(): Promise<DocumentResponse[]> {
    const response = await fetch('/api/documents');
    if (!response.ok) throw new Error('Failed to load documents');
    return response.json();
  },
  async ingestDocument(title: string, content: string): Promise<void> {
    const response = await fetch('/api/documents', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content }),
    });
    if (!response.ok) throw new Error('Failed to ingest document');
  },
  async semanticSearch(query: string): Promise<SearchResultItem[]> {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) throw new Error('Search failed');
    return response.json();
  },
};

export const DocumentIntelligence: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'search' | 'ingest'>('search');

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const data = await apiClient.getDocuments();
      setDocuments(data);
    } catch (err) {
      console.error('Failed to load documents', err);
    }
  };

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) return;

    setLoading(true);
    try {
      await apiClient.ingestDocument(title, content);
      setTitle('');
      setContent('');
      await loadDocuments();
      setActiveTab('search');
    } catch (err) {
      console.error('Ingestion failed', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const results = await apiClient.semanticSearch(searchQuery);
      setSearchResults(results);
    } catch (err) {
      console.error('Search failed', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-slate-900 text-slate-100 min-h-screen rounded-xl shadow-2xl border border-slate-800">
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-indigo-400 flex items-center gap-2">
            <Database className="w-7 h-7" /> AgenticBridge-MCP Document Intelligence
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            PostgreSQL pgvector Semantic Search & Document Orchestration
          </p>
        </div>
        <div className="flex gap-2 bg-slate-800 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'search' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Semantic Search
          </button>
          <button
            onClick={() => setActiveTab('ingest')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'ingest' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Ingest Document
          </button>
        </div>
      </header>

      {activeTab === 'search' ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <form onSubmit={handleSearch} className="flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3.5 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Ask a question or enter a semantic query..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-3 text-sm focus:outline-none focus:border-indigo-500 text-slate-100 placeholder-slate-500"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-3 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                {loading && <Loader2 className="w-4 h-4 animate-spin" />} Search
              </button>
            </form>

            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-slate-200">Search Results</h2>
              {searchResults.length === 0 ? (
                <p className="text-slate-500 text-sm italic bg-slate-800/50 p-6 rounded-lg border border-slate-800 text-center">
                  No search results yet. Enter a query above to query vector embeddings.
                </p>
              ) : (
                searchResults.map((res) => (
                  <div key={res.id} className="bg-slate-800 border border-slate-700/60 rounded-xl p-5 space-y-2">
                    <div className="flex justify-between items-center">
                      <h3 className="font-semibold text-indigo-300">{res.title}</h3>
                      <span className="text-xs bg-indigo-950 text-indigo-300 px-2.5 py-1 rounded-full border border-indigo-800/50 font-mono">
                        Similarity: {(res.similarity_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed">{res.content}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-slate-800/60 border border-slate-800 rounded-xl p-5 space-y-4 h-fit">
            <h2 className="text-md font-semibold text-slate-200 flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" /> Stored Database Chunks ({documents.length})
            </h2>
            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
              {documents.map((doc) => (
                <div key={doc.id} className="bg-slate-900 border border-slate-800 p-3 rounded-lg text-xs space-y-1">
                  <div className="font-medium text-slate-300 truncate">{doc.title}</div>
                  <div className="text-slate-500 truncate">{doc.content}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <form onSubmit={handleIngest} className="max-w-2xl mx-auto space-y-6 bg-slate-800/60 p-8 rounded-xl border border-slate-800">
          <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
            <PlusCircle className="w-5 h-5 text-indigo-400" /> Ingest New Knowledge Chunk
          </h2>
          <div className="space-y-2">
            <label className="text-sm text-slate-300 font-medium">Document Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Enterprise Security Guidelines 2026"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 text-slate-100"
              required
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-slate-300 font-medium">Content Text</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={6}
              placeholder="Enter raw text content to be vectorized and stored..."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-4 text-sm focus:outline-none focus:border-indigo-500 text-slate-100"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-3 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />} Ingest & Generate Embedding
          </button>
        </form>
      )}
    </div>
  );
};