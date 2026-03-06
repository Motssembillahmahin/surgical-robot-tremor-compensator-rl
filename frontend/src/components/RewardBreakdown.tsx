import {
  AreaChart,
  Area,
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

/** Stacked area chart showing all 5 reward components over time. */
export function RewardBreakdown({ history }: Props) {
  const data = history.map((m) => ({
    step: m.step,
    tracking: m.reward_tracking,
    smooth: m.reward_smooth,
    safety: m.reward_safety,
    latency: m.reward_latency,
    human: m.reward_human,
  }));

  return (
    <section aria-label="Reward components breakdown showing tracking, smoothness, safety, latency, and human feedback rewards">
      <h2>Reward Breakdown</h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -5 }} />
          <YAxis label={{ value: "Reward", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          <Area type="monotone" dataKey="tracking" stackId="1" stroke="#3498db" fill="#3498db" name="Tracking" />
          <Area type="monotone" dataKey="smooth" stackId="1" stroke="#2ecc71" fill="#2ecc71" name="Smoothness" />
          <Area type="monotone" dataKey="safety" stackId="1" stroke="#e74c3c" fill="#e74c3c" name="Safety" />
          <Area type="monotone" dataKey="latency" stackId="1" stroke="#f39c12" fill="#f39c12" name="Latency" />
          <Area type="monotone" dataKey="human" stackId="1" stroke="#9b59b6" fill="#9b59b6" name="Human" />
        </AreaChart>
      </ResponsiveContainer>
    </section>
  );
}
