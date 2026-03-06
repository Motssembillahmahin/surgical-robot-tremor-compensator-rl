import { useWebSocket } from "./hooks/useWebSocket";
import { TrajectoryPlot } from "./components/TrajectoryPlot";
import { FrequencySpectrum } from "./components/FrequencySpectrum";
import { RewardBreakdown } from "./components/RewardBreakdown";
import { SafetyZone } from "./components/SafetyZone";
import { TrainingMetricsPanel } from "./components/TrainingMetrics";
import "./App.css";

function App() {
  const { latest, history, connected } = useWebSocket();

  return (
    <div className="dashboard">
      <header>
        <h1>Surgical Tremor Compensator</h1>
        <span className={`status ${connected ? "connected" : "disconnected"}`}>
          {connected ? "Live" : "Disconnected"}
        </span>
        {latest && (
          <span className="meta">
            Step {latest.step.toLocaleString()} | Episode {latest.episode}
          </span>
        )}
      </header>

      <main>
        <TrajectoryPlot history={history} />
        <FrequencySpectrum latest={latest} />
        <RewardBreakdown history={history} />
        <SafetyZone history={history} />
        <TrainingMetricsPanel history={history} />
      </main>
    </div>
  );
}

export default App;
