import type { AgentDetail, AgentSummary, RunDetail, RunEvent, RunSummary } from "@/types/agent";

const BASE = "/api/backend/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  listAgents: (category?: string) =>
    request<AgentSummary[]>(`/agents${category ? `?category=${category}` : ""}`),
  getAgent: (id: string) => request<AgentDetail>(`/agents/${id}`),

  createRun: (agentId: string, message: string) =>
    request<RunSummary>(`/agents/${agentId}/runs`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
  listAgentRuns: (agentId: string) => request<RunSummary[]>(`/agents/${agentId}/runs`),
  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),
  listRunEvents: (runId: string) => request<RunEvent[]>(`/runs/${runId}/events`),
  resumeRun: (runId: string, approved: boolean) =>
    request<RunSummary>(`/runs/${runId}/resume`, {
      method: "POST",
      body: JSON.stringify({ approved }),
    }),

  streamUrl: (runId: string) => `${BASE}/runs/${runId}/stream`,
};
