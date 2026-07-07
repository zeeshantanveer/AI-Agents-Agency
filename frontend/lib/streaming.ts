"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api-client";
import type { RunEvent } from "@/types/agent";

/**
 * Tails a run's SSE stream. Deliberately not built on the Vercel AI SDK's
 * chat protocol — several agents here aren't chat-shaped (Web Researcher
 * returns a report, not a conversation), so a small custom hook that maps
 * backend run_events into whatever the caller needs is more controllable.
 */
export function useRunStream(runId: string | null) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">("idle");
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) return;
    setEvents([]);
    setStatus("streaming");

    const source = new EventSource(api.streamUrl(runId));
    sourceRef.current = source;

    const handle = (evt: MessageEvent) => {
      try {
        const data = JSON.parse(evt.data);
        setEvents((prev) => [...prev, { type: evt.type, ...data }]);
      } catch {
        // ignore keepalive/comment frames
      }
    };

    const eventTypes = [
      "token",
      "node_start",
      "node_end",
      "tool_call",
      "tool_result",
      "interrupt",
      "error",
      "run_complete",
    ];
    for (const type of eventTypes) {
      source.addEventListener(type, handle);
    }

    source.addEventListener("run_complete", () => {
      setStatus("done");
      source.close();
    });
    source.addEventListener("error", () => {
      if (source.readyState === EventSource.CLOSED) {
        setStatus((s) => (s === "streaming" ? "error" : s));
      }
    });

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [runId]);

  return { events, status };
}
