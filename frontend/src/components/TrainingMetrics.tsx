import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetrics as Metrics } from "../types/metrics";

interface Props {
  history: Metrics[];
}

/** Episode return and compensation error with rolling averages. */
export function TrainingMetricsPanel({ history }: Props) {
  // Compute rolling average (window = 20)
  const windowSize = 20;
  const data = history.map((m, i) => {
    const windowStart = Math.max(0, i - windowSize + 1);
    const window = history.slice(windowStart, i + 1);
    const avgReturn = window.reduce((s, x) => s + x.reward_total, 0) / window.length;
    const avgError = window.reduce((s, x) => s + x.compensation_error_mm, 0) / window.length;

    return {
      step: m.step,
      episode: m.episode,
      return: m.reward_total,
      avgReturn: parseFloat(avgReturn.toFixed(4)),
      error: m.compensation_error_mm,
      avgError: parseFloat(avgError.toFixed(4)),
    };
  });

  return (
    <section aria-label="Training metrics showing episode return and compensation error over time with rolling averages">
      <h2>Training Metrics</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -5 }} />
          <YAxis yAxisId="left" label={{ value: "Return", angle: -90, position: "insideLeft" }} />
          <YAxis yAxisId="right" orientation="right" label={{ value: "Error (mm)", angle: 90, position: "insideRight" }} />
          <Tooltip />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="return"
            stroke="#bdc3c7"
            name="Episode Return"
            dot={false}
            strokeWidth={1}
            opacity={0.3}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="avgReturn"
            stroke="#3498db"
            name="Avg Return (20)"
            dot={false}
            strokeWidth={2}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="avgError"
            stroke="#e74c3c"
            name="Avg Error (20)"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
