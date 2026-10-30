import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// --- TypeScript Interfaces ---

export interface DocumentResponse {
  id: number;
  title: string;
  content: string;
  created_at: string;
  updated_at?: string;
}

export interface SearchResultItem {
  id: number;
  title: string;
  content: string;
  similarity_score: number;
}

export interface WorkflowRequestPayload {
  task: string;
  temperature?: number;
}

export interface WorkflowResponse {
  workflow_id: string;
  task: string;
  plan: string[];
  final_output: string;
  execution_logs: string[];
  status: string;
  latency_seconds: number;
}

// --- Configured Axios Instance ---

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for LLM inference workflows
});

// --- API Client Methods ---

export const apiClient = {
  // Document Intelligence & pgvector Endpoints
  async getDocuments(): Promise<DocumentResponse[]> {
    try {
      const response = await axiosInstance.get<DocumentResponse[]>('/documents/');
      return response.data;
    } catch (error) {
      console.error('API Error [getDocuments]:', error);
      throw error;
    }
  },

  async ingestDocument(title: string, content: string): Promise<DocumentResponse> {
    try {
      const response = await axiosInstance.post<DocumentResponse>('/documents/', {
        title,
        content,
      });
      return response.data;
    } catch (error) {
      console.error('API Error [ingestDocument]:', error);
      throw error;
    }
  },

  async semanticSearch(query: string, limit: number = 5): Promise<SearchResultItem[]> {
    try {
      const response = await axiosInstance.post<SearchResultItem[]>('/documents/search', {
        query,
        limit,
      });
      return response.data;
    } catch (error) {
      console.error('API Error [semanticSearch]:', error);
      throw error;
    }
  },

  // LangGraph Agentic Workflow Orchestration Endpoints
  async runWorkflow(task: string, temperature: number = 0.2): Promise<WorkflowResponse> {
    try {
      const payload: WorkflowRequestPayload = { task, temperature };
      const response = await axiosInstance.post<WorkflowResponse>('/api/v1/workflows/', payload);
      return response.data;
    } catch (error) {
      console.error('API Error [runWorkflow]:', error);
      throw error;
    }
  },
};