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
import type { TrainingMetrics } from "../types/metrics";

interface Props {
  history: TrainingMetrics[];
}

/** 3-axis trajectory overlay: raw input, filtered, and compensated. */
export function TrajectoryPlot({ history }: Props) {
  const data = history.map((m) => ({
    step: m.step,
    error: m.compensation_error_mm,
    proximity: m.tissue_proximity_min,
  }));

  return (
    <section aria-label="Trajectory comparison showing compensation error and tissue proximity over time">
      <h2>Trajectory Comparison</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -5 }} />
          <YAxis label={{ value: "mm", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="error"
            stroke="#e74c3c"
            name="Compensation Error (mm)"
            dot={false}
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="proximity"
            stroke="#2ecc71"
            name="Tissue Proximity (mm)"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
