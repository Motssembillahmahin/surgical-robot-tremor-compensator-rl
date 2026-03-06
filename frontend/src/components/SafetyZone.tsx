import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetrics } from "../types/metrics";

interface Props {
  history: TrainingMetrics[];
  safetyMarginMm?: number;
}

/** 2D scatter plot of tissue proximity over time, colour-coded by safety status. */
export function SafetyZone({ history, safetyMarginMm = 2.0 }: Props) {
  const safeData = history
    .filter((m) => m.tissue_proximity_min >= safetyMarginMm * 2)
    .map((m) => ({ step: m.step, proximity: m.tissue_proximity_min }));

  const warningData = history
    .filter((m) => m.tissue_proximity_min >= safetyMarginMm && m.tissue_proximity_min < safetyMarginMm * 2)
    .map((m) => ({ step: m.step, proximity: m.tissue_proximity_min }));

  const dangerData = history
    .filter((m) => m.tissue_proximity_min < safetyMarginMm)
    .map((m) => ({ step: m.step, proximity: m.tissue_proximity_min }));

  return (
    <section aria-label="Safety zone showing robot tip distance to tissue boundary. Green is safe, yellow is warning, red is violation.">
      <h2>Safety Zone</h2>
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" name="Step" type="number" />
          <YAxis dataKey="proximity" name="Distance (mm)" label={{ value: "mm", angle: -90, position: "insideLeft" }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <ReferenceLine y={safetyMarginMm} stroke="#e74c3c" strokeDasharray="5 5" label="Safety Margin" />
          <Scatter name="Safe" data={safeData} fill="#2ecc71" />
          <Scatter name="Warning" data={warningData} fill="#f39c12" />
          <Scatter name="Violation" data={dangerData} fill="#e74c3c" />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  );
}
