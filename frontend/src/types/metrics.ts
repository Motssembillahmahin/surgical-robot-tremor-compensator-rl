/** Live metrics streamed from the training backend via WebSocket. */
export interface TrainingMetrics {
  step: number;
  episode: number;
  reward_total: number;
  reward_tracking: number;
  reward_smooth: number;
  reward_safety: number;
  reward_latency: number;
  reward_human: number;
  compensation_error_mm: number;
  tissue_proximity_min: number;
  sac_entropy: number;
  sac_actor_loss: number;
  sac_critic_loss: number;
  sac_ent_coef: number;
}

/** Trajectory data for a single episode. */
export interface TrajectoryData {
  episode_id: number;
  timestamps_ms: number[];
  raw_input: [number, number, number][];
  filtered_input: [number, number, number][];
  compensated: [number, number, number][];
}

/** Aggregated metrics summary from GET /api/metrics/summary. */
export interface MetricsSummary {
  total_episodes: number;
  total_steps: number;
  best_compensation_error_mm: number;
  rolling_avg_return: number;
  safety_violations_total: number;
}

/** Human feedback submission payload. */
export interface FeedbackRequest {
  episode_id: number;
  score: 1 | 2 | 3 | 4 | 5;
  evaluator_id: string;
}

/** Feedback statistics from GET /feedback/stats. */
export interface FeedbackStats {
  total_labels: number;
  score_distribution: Record<string, number>;
  average_score: number;
}

/** Reward breakdown for chart rendering. */
export interface RewardDataPoint {
  step: number;
  tracking: number;
  smooth: number;
  safety: number;
  latency: number;
  human: number;
  total: number;
}
