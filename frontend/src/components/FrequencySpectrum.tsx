import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetrics } from "../types/metrics";

interface Props {
  latest: TrainingMetrics | null;
}

/**
 * Tremor frequency spectrum placeholder.
 * Full FFT visualisation requires trajectory data from the /api/episodes endpoint.
 * This component shows the current dominant tremor band indicator.
 */
export function FrequencySpectrum({ latest }: Props) {
  if (!latest) {
    return (
      <section aria-label="Tremor frequency spectrum">
        <h2>Frequency Spectrum</h2>
        <p>Waiting for data...</p>
      </section>
    );
  }

  // Placeholder: show tremor band as a bar chart with the 3 profile ranges
  const bands = [
    { name: "Parkinson's", min: 3, max: 6, fill: "#e74c3c" },
    { name: "Essential", min: 4, max: 8, fill: "#f39c12" },
    { name: "Physiological", min: 8, max: 12, fill: "#3498db" },
  ];

  const data = bands.map((b) => ({
    name: b.name,
    range: b.max - b.min,
    min: b.min,
  }));

  return (
    <section aria-label="Tremor frequency spectrum showing pathological frequency bands">
      <h2>Frequency Spectrum</h2>
      <p>
        SAC Entropy: <strong>{latest.sac_entropy.toFixed(3)}</strong> |
        Ent Coef: <strong>{latest.sac_ent_coef.toFixed(4)}</strong>
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis label={{ value: "Hz", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Bar dataKey="range" fill="#3498db" name="Frequency Range (Hz)" />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
