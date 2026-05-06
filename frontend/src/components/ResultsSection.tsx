import { FiClipboard, FiDownload, FiFileText } from "react-icons/fi";
import type { AnalysisRecord } from "../store/analysisStore";
import { formatDateTime, formatPercent } from "../utils/formatters";

interface ResultsSectionProps {
  analysis: AnalysisRecord;
  onCopy: () => Promise<void>;
  onExportJson: () => void;
  onExportCsv: () => void;
}

const ConfidenceGauge = ({ value }: { value: number }): JSX.Element => {
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.max(0, Math.min(1, value));
  const dashOffset = circumference * (1 - progress);

  return (
    <div className="relative flex h-28 w-28 items-center justify-center">
      <svg className="h-28 w-28 -rotate-90" viewBox="0 0 120 120" role="img" aria-label="Confidence gauge">
        <circle cx="60" cy="60" r={radius} stroke="rgba(255,255,255,0.1)" strokeWidth="10" fill="transparent" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          stroke="url(#confidenceGradient)"
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          fill="transparent"
        />
        <defs>
          <linearGradient id="confidenceGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#00D9FF" />
            <stop offset="100%" stopColor="#D700FF" />
          </linearGradient>
        </defs>
      </svg>
      <span className="absolute text-lg font-semibold text-text-primary">{Math.round(progress * 100)}%</span>
    </div>
  );
};

export const ResultsSection = ({ analysis, onCopy, onExportJson, onExportCsv }: ResultsSectionProps): JSX.Element => {
  const isFake = analysis.prediction === 1;
  const fakeProbability = isFake ? analysis.confidence : 1 - analysis.confidence;
  const realProbability = 1 - fakeProbability;

  return (
    <section className="glow-purple animate-fade-in rounded-2xl border border-neon-cyan/30 bg-gradient-card p-6 shadow-2xl backdrop-blur-md transition-all duration-300 hover:border-neon-cyan/60">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-text-primary">Detection Result</h2>
        <span
          className={`rounded-full px-4 py-1 text-sm font-semibold text-text-primary ${
            isFake ? "bg-gradient-danger" : "bg-gradient-success"
          }`}
        >
          {isFake ? "FAKE" : "REAL"}
        </span>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <article className="rounded-2xl border border-neon-cyan/30 bg-gradient-card p-6 shadow-2xl backdrop-blur-md transition-all duration-300 hover:border-neon-cyan/60">
          <h3 className="mb-3 text-sm font-medium text-text-muted">Confidence</h3>
          <ConfidenceGauge value={analysis.confidence} />
        </article>

        <article className="rounded-2xl border border-neon-cyan/30 bg-gradient-card p-6 shadow-2xl backdrop-blur-md transition-all duration-300 hover:border-neon-cyan/60">
          <h3 className="mb-3 text-sm font-medium text-text-muted">Probability Breakdown</h3>
          <div className="space-y-2 text-sm text-text-primary">
            <p>Fake probability: {formatPercent(fakeProbability)}</p>
            <p>Real probability: {formatPercent(realProbability)}</p>
            <p>Detected language: {analysis.language === "hi" ? "Hindi" : "English"}</p>
            <p>Timestamp: {formatDateTime(analysis.createdAt)}</p>
          </div>
        </article>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => void onCopy()}
          className="inline-flex items-center gap-2 rounded-lg border border-neon-primary/60 px-3 py-2 text-sm text-neon-primary transition hover:bg-neon-primary/10"
        >
          <FiClipboard /> Copy
        </button>
        <button
          type="button"
          onClick={onExportJson}
          className="inline-flex items-center gap-2 rounded-lg border border-white/25 px-3 py-2 text-sm text-text-primary transition hover:bg-white/10"
        >
          <FiFileText /> Export JSON
        </button>
        <button
          type="button"
          onClick={onExportCsv}
          className="inline-flex items-center gap-2 rounded-lg border border-white/25 px-3 py-2 text-sm text-text-primary transition hover:bg-white/10"
        >
          <FiDownload /> Export CSV
        </button>
      </div>
    </section>
  );
};
