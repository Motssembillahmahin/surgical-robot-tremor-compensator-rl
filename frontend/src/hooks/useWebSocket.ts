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
        const data: TrainingMetrics = JSON.parse(event.data);
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
