import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiClient } from './api';
import axios from 'axios';

// Mock axios module globally
vi.mock('axios', () => {
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
  };
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance),
    },
  };
});

describe('apiClient Service - Comprehensive Test Suite', () => {
  let mockAxiosInstance: any;

  beforeEach(() => {
    vi.clearAllMocks();
    mockAxiosInstance = (axios.create as any)();
  });

  // --- 1. Document Management & Retrieval ---

  it('should fetch documents successfully', async () => {
    const mockDocuments = [
      { id: 1, title: 'Threat Intel Report', content: 'Sample text', created_at: '2026-10-30T12:00:00Z' },
      { id: 2, title: 'Policy Compliance', content: 'Mandate guidelines', created_at: '2026-10-30T12:05:00Z' },
    ];

    mockAxiosInstance.get.mockResolvedValueOnce({ data: mockDocuments });

    const result = await apiClient.getDocuments();

    expect(mockAxiosInstance.get).toHaveBeenCalledTimes(1);
    expect(mockAxiosInstance.get).toHaveBeenCalledWith('/documents/');
    expect(result).toEqual(mockDocuments);
  });

  it('should handle errors gracefully when fetching documents fails', async () => {
    const networkError = new Error('Network Error 500');
    mockAxiosInstance.get.mockRejectedValueOnce(networkError);

    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(apiClient.getDocuments()).rejects.toThrow('Network Error 500');
    expect(consoleSpy).toHaveBeenCalledWith('API Error [getDocuments]:', networkError);

    consoleSpy.mockRestore();
  });

  // --- 2. Document Ingestion ---

  it('should ingest a document successfully', async () => {
    const title = 'Cybersecurity Threat Bulletin';
    const content = 'Unusual outbound traffic detected across regional subnets.';
    const mockResponse = { id: 10, title, content, created_at: '2026-10-30T12:10:00Z' };

    mockAxiosInstance.post.mockResolvedValueOnce({ data: mockResponse });

    const result = await apiClient.ingestDocument(title, content);

    expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/documents/', { title, content });
    expect(result).toEqual(mockResponse);
  });

  it('should handle errors when document ingestion fails', async () => {
    const serverError = new Error('Validation Error 422');
    mockAxiosInstance.post.mockRejectedValueOnce(serverError);

    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(apiClient.ingestDocument('', '')).rejects.toThrow('Validation Error 422');
    expect(consoleSpy).toHaveBeenCalledWith('API Error [ingestDocument]:', serverError);

    consoleSpy.mockRestore();
  });

  // --- 3. Semantic Search via pgvector ---

  it('should perform semantic search successfully with default limit', async () => {
    const query = 'phishing detection algorithms';
    const mockSearchResults = [
      { id: 3, title: 'Email Gateway Logs', content: 'Phishing signature matched...', similarity_score: 0.94 },
    ];

    mockAxiosInstance.post.mockResolvedValueOnce({ data: mockSearchResults });

    const result = await apiClient.semanticSearch(query);

    expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/documents/search', {
      query,
      limit: 5,
    });
    expect(result).toEqual(mockSearchResults);
  });

  it('should perform semantic search with a custom limit', async () => {
    const query = 'stock ownership mandate';
    const limit = 10;
    const mockSearchResults = [];

    mockAxiosInstance.post.mockResolvedValueOnce({ data: mockSearchResults });

    const result = await apiClient.semanticSearch(query, limit);

    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/documents/search', {
      query,
      limit,
    });
    expect(result).toEqual(mockSearchResults);
  });

  it('should handle errors when semantic search fails', async () => {
    const searchError = new Error('pgvector Index Timeout');
    mockAxiosInstance.post.mockRejectedValueOnce(searchError);

    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(apiClient.semanticSearch('malware analysis')).rejects.toThrow('pgvector Index Timeout');
    expect(consoleSpy).toHaveBeenCalledWith('API Error [semanticSearch]:', searchError);

    consoleSpy.mockRestore();
  });

  // --- 4. LangGraph Agentic Workflow Orchestration ---

  it('should trigger an agentic workflow successfully with default temperature', async () => {
    const task = 'Synthesize regional threat telemetry';
    const mockWorkflowResponse = {
      workflow_id: 'wf_exec_9876',
      task,
      plan: ['Step 1: Intake', 'Step 2: Analysis'],
      final_output: 'Threat synthesis complete across target sectors.',
      execution_logs: ['[12:00:01] Initialized'],
      status: 'success',
      latency_seconds: 1.45,
    };

    mockAxiosInstance.post.mockResolvedValueOnce({ data: mockWorkflowResponse });

    const result = await apiClient.runWorkflow(task);

    expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/workflows/', {
      task,
      temperature: 0.2,
    });
    expect(result).toEqual(mockWorkflowResponse);
  });

  it('should trigger an agentic workflow with custom temperature', async () => {
    const task = 'Generate automated security incident response policy';
    const temperature = 0.7;
    const mockWorkflowResponse = {
      workflow_id: 'wf_exec_5432',
      task,
      plan: ['Step 1: Draft policy'],
      final_output: 'Policy generated.',
      execution_logs: [],
      status: 'success',
      latency_seconds: 2.1,
    };

    mockAxiosInstance.post.mockResolvedValueOnce({ data: mockWorkflowResponse });

    const result = await apiClient.runWorkflow(task, temperature);

    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/api/v1/workflows/', {
      task,
      temperature,
    });
    expect(result).toEqual(mockWorkflowResponse);
  });

  it('should handle errors when workflow execution fails', async () => {
    const workflowError = new Error('OpenAI Rate Limit Exceeded');
    mockAxiosInstance.post.mockRejectedValueOnce(workflowError);

    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(apiClient.runWorkflow('Complex agent loop')).rejects.toThrow('OpenAI Rate Limit Exceeded');
    expect(consoleSpy).toHaveBeenCalledWith('API Error [runWorkflow]:', workflowError);

    consoleSpy.mockRestore();
  });
});