export interface AgentSummary {
  id: string;
  slug: string;
  name: string;
  description: string;
  category: "developer" | "research" | "business_ops" | "custom";
  origin: "built_in" | "generated";
  version: string;
  is_active: boolean;
}

export interface AgentDetail extends AgentSummary {
  spec: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export type RunStatus = "queued" | "running" | "waiting_approval" | "succeeded" | "failed" | "cancelled";

export interface RunSummary {
  id: string;
  agent_id: string;
  status: RunStatus;
  cost_usd: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface RunDetail extends RunSummary {
  input: { message: string };
  output: { text?: string; terminal_reason?: string; interrupt?: RunInterrupt } | null;
  error: string | null;
}

export interface RunInterrupt {
  type: "tool_approval" | "human_review";
  tool_id?: string;
  tool_name?: string;
  args?: Record<string, unknown>;
  stage?: string;
  node_id?: string;
  content?: string;
}

export interface RunEvent {
  seq?: number;
  type: string;
  [key: string]: unknown;
}
