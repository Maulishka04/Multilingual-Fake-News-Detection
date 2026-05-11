import { useMemo, useState } from "react";
import { FiDownload, FiFileText, FiTrash2, FiX } from "react-icons/fi";
import { toast } from "react-hot-toast";
import { useHistory } from "../hooks/useHistory";
import type { AnalysisRecord } from "../store/analysisStore";
import { formatDateTime, truncateText } from "../utils/formatters";

interface HistorySidebarProps {
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  onSelectAnalysis: (analysis: AnalysisRecord) => void;
}

export const HistorySidebar = ({
  isMobileOpen,
  onCloseMobile,
  onSelectAnalysis,
}: HistorySidebarProps): JSX.Element => {
  const { history, deleteAnalysis, clearHistory, exportJSON, exportCSV } = useHistory();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "fake" | "real">("all");
  const [showConfirm, setShowConfirm] = useState(false);

  const filteredHistory = useMemo(
    () =>
      history.filter((item) => {
        const byText = item.text.toLowerCase().includes(query.toLowerCase());
        const byPrediction =
          filter === "all" ||
          (filter === "fake" && item.prediction === 1) ||
          (filter === "real" && item.prediction === 0);
        return byText && byPrediction;
      }),
    [filter, history, query]
  );

  const clearAllWithConfirmation = (): void => {
    clearHistory();
    setShowConfirm(false);
    toast.success("History cleared", { duration: 3000 });
  };

  const exportAllJson = (): void => {
    exportJSON();
    toast.success("Exported successfully", { duration: 3000 });
  };

  const exportAllCsv = (): void => {
    exportCSV();
    toast.success("Exported successfully", { duration: 3000 });
  };

  return (
    <>
      <aside
        className={`fixed inset-y-0 right-0 z-40 w-full max-w-sm transform rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60 md:static md:z-auto md:max-w-none md:translate-x-0 ${
          isMobileOpen ? "translate-x-0" : "translate-x-full"
        }`}
        aria-label="Analysis history"
      >
        <header className="mb-4 flex items-center justify-between gap-2">
          <h2 className="text-lg font-semibold font-display text-[var(--color-text-primary)]">History</h2>
          <button
            type="button"
            className="rounded-lg border border-[var(--color-border)] p-2 text-[var(--color-text-primary)] md:hidden"
            onClick={onCloseMobile}
            aria-label="Close history sidebar"
          >
            <FiX />
          </button>
        </header>

        <div className="mb-3 space-y-2">
          <label className="sr-only" htmlFor="history-search">
            Search history
          </label>
          <input
            id="history-search"
            type="search"
            placeholder="Search text..."
            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/60"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <select
            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-sm text-[var(--color-text-primary)]"
            value={filter}
            onChange={(event) => setFilter(event.target.value as "all" | "fake" | "real")}
            aria-label="Filter history by prediction"
          >
            <option value="all">All</option>
            <option value="fake">Fake</option>
            <option value="real">Real</option>
          </select>
        </div>

        <div className="mb-3 flex flex-wrap gap-2 text-xs">
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] px-2 py-1 text-[var(--color-text-primary)] hover:bg-[var(--color-primary)]/10 transition duration-300"
            onClick={exportAllJson}
          >
            <FiFileText /> JSON
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] px-2 py-1 text-[var(--color-text-primary)] hover:bg-[var(--color-primary)]/10 transition duration-300"
            onClick={exportAllCsv}
          >
            <FiDownload /> CSV
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border border-[var(--color-danger)]/50 px-2 py-1 text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10 transition duration-300"
            onClick={() => setShowConfirm(true)}
          >
            <FiTrash2 /> Clear all
          </button>
        </div>

        <div className="max-h-[55vh] space-y-2 overflow-y-auto pr-1 md:max-h-[70vh]">
          {filteredHistory.length === 0 ? (
            <p className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] p-3 text-sm text-[var(--color-text-secondary)]">No history</p>
          ) : (
            filteredHistory.map((item) => (
              <article
                key={item.id}
                className="group cursor-pointer rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow transition-all duration-300 hover:border-[var(--color-primary)]/60 hover:shadow-xl"
                onClick={() => onSelectAnalysis(item)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    onSelectAnalysis(item);
                  }
                }}
                role="button"
                tabIndex={0}
                aria-label="View historical analysis"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm text-[var(--color-text-primary)]">{truncateText(item.text, 50)}</p>
                    <p className="mt-1 text-xs text-[var(--color-text-secondary)]">{formatDateTime(item.createdAt)}</p>
                  </div>
                  <button
                    type="button"
                    className="invisible rounded-md border border-[var(--color-border)] p-1 text-[var(--color-text-secondary)] group-hover:visible"
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteAnalysis(item.id);
                      toast.success("Entry removed", { duration: 3000 });
                    }}
                    aria-label="Delete history item"
                  >
                    <FiTrash2 />
                  </button>
                </div>
                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className={item.prediction === 1 ? "text-[var(--color-danger)] font-semibold" : "text-[var(--color-success)] font-semibold"}>
                    {item.prediction === 1 ? "Fake" : "Real"}
                  </span>
                  <span className="text-[var(--color-text-secondary)]">{Math.round(item.confidence * 100)}%</span>
                </div>
              </article>
            ))
          )}
        </div>
      </aside>

      {isMobileOpen ? (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      ) : null}

      {showConfirm ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-sm rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">Clear history?</h3>
            <p className="mt-2 text-sm text-[var(--color-text-secondary)]">This will remove all saved analyses.</p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                className="rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-text-primary)] hover:bg-[var(--color-bg-primary)] transition duration-300"
                onClick={() => setShowConfirm(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="rounded-lg border border-[var(--color-danger)]/60 px-3 py-2 text-sm text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10 transition duration-300"
                onClick={clearAllWithConfirmation}
              >
                Clear all
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
};
