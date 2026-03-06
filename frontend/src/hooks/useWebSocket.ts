import { useEffect, useRef, useState, useCallback } from "react";
import type { TrainingMetrics } from "../types/metrics";

const DEFAULT_WS_URL = "ws://localhost:8000/ws/metrics";
const RECONNECT_DELAY_MS = 3000;
const MAX_HISTORY = 500;

interface UseWebSocketReturn {
  /** Most recent metrics snapshot. */
  latest: TrainingMetrics | null;
  /** Rolling history of metrics for time-series charts. */
  history: TrainingMetrics[];
  /** WebSocket connection state. */
  connected: boolean;
}

/**
 * WebSocket hook that connects to the training backend and streams
 * live metrics. Automatically reconnects on disconnect.
 */
export function useWebSocket(url: string = DEFAULT_WS_URL): UseWebSocketReturn {
  const [latest, setLatest] = useState<TrainingMetrics | null>(null);
  const [history, setHistory] = useState<TrainingMetrics[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        // Ignore pings and status messages
        if (raw.type === "ping" || raw.type === "status") return;
        // Only process metrics messages
        if (raw.type !== "metrics") return;

        const data: TrainingMetrics = {
          step: raw.step ?? 0,
          episode: raw.episode ?? 0,
          reward_total: raw.reward_total ?? 0,
          reward_tracking: raw.reward_tracking ?? 0,
          reward_smooth: raw.reward_smooth ?? 0,
          reward_safety: raw.reward_safety ?? 0,
          reward_latency: raw.reward_latency ?? 0,
          reward_human: raw.reward_human ?? 0,
          compensation_error_mm: raw.compensation_error_mm ?? 0,
          tissue_proximity_min: raw.tissue_proximity_min ?? 50,
          sac_entropy: raw.sac_entropy ?? 0,
          sac_actor_loss: raw.sac_actor_loss ?? 0,
          sac_critic_loss: raw.sac_critic_loss ?? 0,
          sac_ent_coef: raw.sac_ent_coef ?? 0,
        };
        setLatest(data);
        setHistory((prev) => {
          const next = [...prev, data];
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
        });
      } catch {
        console.warn("Failed to parse WebSocket message:", event.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  return { latest, history, connected };
}
