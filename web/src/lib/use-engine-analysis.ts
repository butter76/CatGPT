"use client";

import { useState, useCallback, useRef } from "react";
import type { EngineInfoLine, EngineKind, CatGPTSearchStats } from "./types";

export interface EngineAnalysisState {
  /** Whether an analysis is currently running */
  running: boolean;
  /** Current engine being used */
  engine: EngineKind | null;
  /** All depth info lines received so far (UCI engines) */
  depthHistory: EngineInfoLine[];
  /** The latest info line (UCI engines) */
  latestInfo: EngineInfoLine | null;
  /** The final bestmove (set when analysis completes) */
  bestMove: string | null;
  /** Ponder move */
  ponder: string | null;
  /** Whether the result was saved to DB */
  saved: boolean;
  /** Error message if something went wrong */
  error: string | null;
  /** Latest CatGPT search stats (catgpt engine only) */
  catgptStats: CatGPTSearchStats | null;
  /** All CatGPT stats events received so far */
  catgptHistory: CatGPTSearchStats[];
}

const INITIAL_STATE: EngineAnalysisState = {
  running: false,
  engine: null,
  depthHistory: [],
  latestInfo: null,
  bestMove: null,
  ponder: null,
  saved: false,
  error: null,
  catgptStats: null,
  catgptHistory: [],
};

export function useEngineAnalysis() {
  const [state, setState] = useState<EngineAnalysisState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const startAnalysis = useCallback(
    async (params: {
      fen: string;
      engine: EngineKind;
      nodes: number;
      positionId?: string;
    }) => {
      // Cancel any running analysis
      if (abortRef.current) {
        abortRef.current.abort();
      }

      const controller = new AbortController();
      abortRef.current = controller;

      setState({
        ...INITIAL_STATE,
        running: true,
        engine: params.engine,
      });

      try {
        const url = new URL("/api/analyze/live", window.location.origin);
        url.searchParams.set("fen", params.fen);
        url.searchParams.set("engine", params.engine);
        url.searchParams.set("nodes", params.nodes.toString());
        if (params.positionId) {
          url.searchParams.set("positionId", params.positionId);
        }

        const response = await fetch(url.toString(), {
          signal: controller.signal,
        });

        if (!response.ok) {
          const err = await response.json();
          setState((s) => ({
            ...s,
            running: false,
            error: err.error || "Request failed",
          }));
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          setState((s) => ({
            ...s,
            running: false,
            error: "No response body",
          }));
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events from buffer
          const events = buffer.split("\n\n");
          buffer = events.pop() || ""; // Keep incomplete event in buffer

          for (const eventBlock of events) {
            if (!eventBlock.trim()) continue;

            let eventType = "";
            let eventData = "";

            for (const line of eventBlock.split("\n")) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7);
              } else if (line.startsWith("data: ")) {
                eventData = line.slice(6);
              }
            }

            if (!eventType || !eventData) continue;

            try {
              const parsed = JSON.parse(eventData);

              switch (eventType) {
                case "info":
                  setState((s) => ({
                    ...s,
                    depthHistory: [...s.depthHistory, parsed as EngineInfoLine],
                    latestInfo: parsed as EngineInfoLine,
                  }));
                  break;

                case "catgpt_stats":
                  setState((s) => ({
                    ...s,
                    catgptStats: parsed as CatGPTSearchStats,
                    catgptHistory: [...s.catgptHistory, parsed as CatGPTSearchStats],
                  }));
                  break;

                case "bestmove":
                  setState((s) => ({
                    ...s,
                    bestMove: parsed.bestMove,
                    ponder: parsed.ponder ?? null,
                  }));
                  break;

                case "saved":
                  setState((s) => ({ ...s, saved: true }));
                  break;

                case "error":
                  setState((s) => ({
                    ...s,
                    error: parsed.message,
                  }));
                  break;

                case "done":
                  setState((s) => ({ ...s, running: false }));
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }

        // Stream ended
        setState((s) => ({ ...s, running: false }));
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          setState((s) => ({ ...s, running: false }));
          return;
        }
        setState((s) => ({
          ...s,
          running: false,
          error: err instanceof Error ? err.message : String(err),
        }));
      }
    },
    []
  );

  const stopAnalysis = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setState((s) => ({ ...s, running: false }));
  }, []);

  const reset = useCallback(() => {
    stopAnalysis();
    setState(INITIAL_STATE);
  }, [stopAnalysis]);

  return { ...state, startAnalysis, stopAnalysis, reset };
}
