import { useState } from "react";
import type { TrainingMetrics } from "../types/metrics";

interface Props {
  latest: TrainingMetrics | null;
}

interface FeedbackStats {
  total_labels: number;
  score_distribution: Record<string, number>;
  average_score: number;
  evaluator_count: number;
}

/** Panel for submitting human feedback scores and viewing statistics. */
export function FeedbackPanel({ latest }: Props) {
  const [score, setScore] = useState<number>(3);
  const [evaluatorId, setEvaluatorId] = useState("evaluator_1");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [stats, setStats] = useState<FeedbackStats | null>(null);

  const episodeId = latest?.episode ?? 0;

  const submitFeedback = async () => {
    setSubmitting(true);
    setMessage(null);
    try {
      const res = await fetch("/feedback/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          episode_id: episodeId,
          score,
          evaluator_id: evaluatorId,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setMessage(`Submitted! Total labels: ${data.total_labels}`);
      } else {
        setMessage(`Error: ${res.statusText}`);
      }
    } catch (err) {
      setMessage("Failed to submit feedback");
    }
    setSubmitting(false);
  };

  const loadStats = async () => {
    try {
      const res = await fetch("/feedback/stats");
      if (res.ok) {
        setStats(await res.json());
      }
    } catch {
      // ignore
    }
  };

  const scoreLabels = [
    "1 - Severe issues",
    "2 - Poor",
    "3 - Acceptable",
    "4 - Good",
    "5 - Excellent",
  ];

  return (
    <section aria-label="Human feedback submission and statistics panel">
      <h2>Human Feedback</h2>

      <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
        <div>
          <label style={{ display: "block", marginBottom: "0.5rem", fontSize: "0.85rem" }}>
            Episode: <strong>{episodeId}</strong>
          </label>

          <label style={{ display: "block", marginBottom: "0.5rem", fontSize: "0.85rem" }}>
            Evaluator ID:
            <input
              type="text"
              value={evaluatorId}
              onChange={(e) => setEvaluatorId(e.target.value)}
              style={{ marginLeft: "0.5rem", padding: "0.2rem 0.4rem", width: "120px" }}
            />
          </label>

          <fieldset style={{ border: "none", padding: 0, margin: "0.5rem 0" }}>
            <legend style={{ fontSize: "0.85rem", marginBottom: "0.3rem" }}>
              Compensation Quality:
            </legend>
            {scoreLabels.map((label, i) => (
              <label key={i + 1} style={{ display: "block", fontSize: "0.8rem", cursor: "pointer" }}>
                <input
                  type="radio"
                  name="score"
                  value={i + 1}
                  checked={score === i + 1}
                  onChange={() => setScore(i + 1)}
                />
                {" "}{label}
              </label>
            ))}
          </fieldset>

          <button
            onClick={submitFeedback}
            disabled={submitting || !latest}
            style={{
              padding: "0.4rem 1rem",
              background: "#3498db",
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              cursor: submitting ? "wait" : "pointer",
              marginTop: "0.5rem",
            }}
          >
            {submitting ? "Submitting..." : "Submit Score"}
          </button>

          {message && (
            <p style={{ fontSize: "0.8rem", color: "#27ae60", marginTop: "0.3rem" }}>
              {message}
            </p>
          )}
        </div>

        <div>
          <button
            onClick={loadStats}
            style={{
              padding: "0.3rem 0.8rem",
              background: "#ecf0f1",
              border: "1px solid #bdc3c7",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Load Stats
          </button>

          {stats && (
            <div style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
              <p>Total labels: <strong>{stats.total_labels}</strong></p>
              <p>Average score: <strong>{stats.average_score.toFixed(2)}</strong></p>
              <p>Evaluators: <strong>{stats.evaluator_count}</strong></p>
              <div style={{ display: "flex", gap: "0.3rem", marginTop: "0.3rem" }}>
                {[1, 2, 3, 4, 5].map((s) => (
                  <span
                    key={s}
                    style={{
                      padding: "0.2rem 0.4rem",
                      background: s <= 2 ? "#fadbd8" : s >= 4 ? "#d5f5e3" : "#fdebd0",
                      borderRadius: "3px",
                    }}
                  >
                    {s}: {stats.score_distribution[String(s)] ?? 0}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
