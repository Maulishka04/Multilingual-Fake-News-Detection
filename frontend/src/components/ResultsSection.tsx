import { FiClipboard, FiDownload, FiFileText } from "react-icons/fi";
import type { AnalysisRecord } from "../store/analysisStore";
import { formatDateTime, formatPercent } from "../utils/formatters";
import { MODEL_INFO, type ModelType } from "../utils/constants";

interface ResultsSectionProps {
  analysis: AnalysisRecord;
  onCopy: () => Promise<void>;
  onExportJson: () => void;
  onExportCsv: () => void;
  modelType?: "svm" | "mbert"; // Add modelType prop
}

const ConfidenceGauge = ({ value }: { value: number }): JSX.Element => {
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.max(0, Math.min(1, value));
  const dashOffset = circumference * (1 - progress);

  return (
    <div className="relative flex h-28 w-28 items-center justify-center">
      <svg className="h-28 w-28 -rotate-90" viewBox="0 0 120 120" role="img" aria-label="Confidence gauge">
        <circle cx="60" cy="60" r={radius} stroke="var(--color-text-primary)" strokeWidth="10" strokeOpacity="0.08" fill="transparent" />
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
            <stop offset="0%" stopColor="#00A19B" />
            <stop offset="100%" stopColor="#CCDA47" />
          </linearGradient>
        </defs>
      </svg>
      <span className="absolute text-lg font-semibold text-[var(--color-text-primary)]">{Math.round(progress * 100)}%</span>
    </div>
  );
};

export const ResultsSection = ({
  analysis,
  onCopy,
  onExportJson,
  onExportCsv,
  modelType = "svm", // Default to "svm"
}: ResultsSectionProps): JSX.Element => {
  const isFake = analysis.prediction === 1;
  const fakeProbability = isFake ? analysis.confidence : 1 - analysis.confidence;
  const realProbability = 1 - fakeProbability;

  return (
    <section className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow-lg transition-all duration-300 hover:border-[var(--color-primary)]/60">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold font-display text-[var(--color-text-primary)]">Detection Result</h2>
        <span
          className={`rounded-full px-4 py-1 text-sm font-semibold ${isFake ? "bg-[var(--color-danger)] text-white" : "bg-[var(--color-success)] text-white"}`}
        >
          {isFake ? "FAKE" : "REAL"}
        </span>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <article className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
          <h3 className="mb-3 text-sm font-medium text-[var(--color-text-secondary)]">Confidence</h3>
          <ConfidenceGauge value={analysis.confidence} />
        </article>

        <article className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
          <h3 className="mb-3 text-sm font-medium text-[var(--color-text-secondary)]">Probability Breakdown</h3>
          <div className="space-y-2 text-sm text-[var(--color-text-primary)]">
            <p>Fake probability: {formatPercent(fakeProbability)}</p>
            <p>Real probability: {formatPercent(realProbability)}</p>
            <p>Detected language: {analysis.language === "hi" ? "Hindi" : "English"}</p>
            <p>Timestamp: {formatDateTime(analysis.createdAt)}</p>
          </div>
        </article>

        <article className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
          <h3 className="mb-3 text-sm font-medium text-[var(--color-text-secondary)]">Model Details</h3>
          <div className="space-y-2 text-sm text-[var(--color-text-primary)]">
            <p><strong>Model:</strong> {MODEL_INFO[modelType as ModelType].name}</p>
            <p><strong>Accuracy:</strong> {MODEL_INFO[modelType as ModelType].accuracy}</p>
            <p><strong>F1-Score:</strong> {MODEL_INFO[modelType as ModelType].f1Score}</p>
            <p><strong>Inference Time:</strong> {MODEL_INFO[modelType as ModelType].inferenceTime}</p>
          </div>
        </article>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => void onCopy()}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-primary)]/60 px-3 py-2 text-sm text-[var(--color-primary)] transition duration-300 hover:bg-[var(--color-primary)]/10"
        >
          <FiClipboard /> Copy
        </button>
        <button
          type="button"
          onClick={onExportJson}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-text-primary)] transition duration-300 hover:bg-[var(--color-bg-secondary)]"
        >
          <FiFileText /> Export JSON
        </button>
        <button
          type="button"
          onClick={onExportCsv}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-text-primary)] transition duration-300 hover:bg-[var(--color-bg-secondary)]"
        >
          <FiDownload /> Export CSV
        </button>
      </div>
    </section>
  );
};
