import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplanationPayload } from "../services/api";

interface ExplainerVisualizationProps {
  explanation: ExplanationPayload | null;
}

interface ChartRow {
  word: string;
  score: number;
}

export const ExplainerVisualization = ({ explanation }: ExplainerVisualizationProps): JSX.Element => {
  if (!explanation) {
    return (
      <section className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
        <h2 className="text-lg font-semibold font-display text-[var(--color-text-primary)]">LIME Explanation</h2>
        <p className="mt-2 text-sm text-[var(--color-text-secondary)]">No explanation available.</p>
      </section>
    );
  }

  const chartData: ChartRow[] = Object.entries(explanation.word_scores)
    .map(([word, score]) => ({ word, score }))
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, 10)
    .reverse();

  return (
    <section className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
      <h2 className="mb-4 text-lg font-semibold font-display text-[var(--color-text-primary)]">LIME Explanation</h2>

      <div className="grid gap-4 md:grid-cols-2">
        <article className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
          <h3 className="mb-2 text-sm font-medium text-[var(--color-success)]">Positive Words</h3>
          <div className="flex flex-wrap gap-2">
            {explanation.positive_words.length ? (
              explanation.positive_words.map((word) => (
                <span key={word} className="rounded-full bg-[var(--color-success)]/20 px-2 py-1 text-xs text-[var(--color-success)]">
                  {word}
                </span>
              ))
            ) : (
              <span className="text-sm text-text-muted">No positive indicators.</span>
            )}
          </div>
        </article>

        <article className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
          <h3 className="mb-2 text-sm font-medium text-[var(--color-danger)]">Negative Words</h3>
          <div className="flex flex-wrap gap-2">
            {explanation.negative_words.length ? (
              explanation.negative_words.map((word) => (
                <span key={word} className="rounded-full bg-[var(--color-danger)]/20 px-2 py-1 text-xs text-[var(--color-danger)]">
                  {word}
                </span>
              ))
            ) : (
              <span className="text-sm text-text-muted">No negative indicators.</span>
            )}
          </div>
        </article>
      </div>

      <article className="mt-4 rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
        <h3 className="mb-3 text-sm font-medium text-[var(--color-text-secondary)]">Word Importance (Top 10)</h3>
        {chartData.length ? (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 12, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,161,155,0.2)" />
                <XAxis type="number" stroke="var(--color-text-primary)" tick={{ fill: "var(--color-text-primary)" }} />
                <YAxis type="category" dataKey="word" stroke="var(--color-text-primary)" tick={{ fill: "var(--color-text-primary)" }} width={100} />
                <Tooltip
                  contentStyle={{ background: "var(--color-bg-primary)", border: "1px solid var(--color-border)", color: "var(--color-text-primary)" }}
                  formatter={(value) => [Number(value).toFixed(4), "Importance"]}
                />
                <Bar dataKey="score">
                  {chartData.map((entry) => (
                    <Cell key={entry.word} fill={entry.score >= 0 ? "var(--color-success)" : "var(--color-danger)"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-secondary)]">No explanation scores available.</p>
        )}
      </article>

      <article className="mt-4 rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
        <h3 className="mb-2 text-sm font-medium text-[var(--color-text-secondary)]">Full Explanation</h3>
        <p className="text-sm text-[var(--color-text-primary)]">{explanation.explanation_text || "No explanation available."}</p>
      </article>
    </section>
  );
};
