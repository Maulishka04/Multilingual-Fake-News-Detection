import { useCallback, useEffect, useMemo, useState, Suspense, lazy } from "react";
import { Toaster, toast } from "react-hot-toast";
import { FiClock } from "react-icons/fi";
import "./styles/globals.css";
import "./styles/colors.css";
import "./styles/typography.css";
import "./styles/animations.css";
const ExplainerVisualization = lazy(() => import("./components/ExplainerVisualization").then((mod) => ({ default: mod.ExplainerVisualization })));
import { HistorySidebar } from "./components/HistorySidebar";
import { InputSection } from "./components/InputSection";
import { ExplainerSkeleton, ResultsSkeleton } from "./components/LoadingSkeletons";
const ResultsSection = lazy(() => import("./components/ResultsSection").then((mod) => ({ default: mod.ResultsSection })));
import WelcomeScreen from "./components/WelcomeScreen";
import { useWelcomeScreen } from "./hooks/useWelcomeScreen";
import { useAnalysis } from "./hooks/useAnalysis";
import { useHistory } from "./hooks/useHistory";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useAnalysisStore, type AnalysisRecord } from "./store/analysisStore";
import { type LanguageMode, type ModelType } from "./utils/constants";
import { copyTextToClipboard, exportToCsv, exportToJson } from "./utils/exporters";

const toCsvRows = (analyses: AnalysisRecord[]): Array<Record<string, string | number>> =>
  analyses.map((item) => ({
    id: item.id,
    text: item.text,
    language: item.language,
    prediction: item.prediction,
    confidence: Number((item.confidence * 100).toFixed(2)),
    createdAt: item.createdAt,
    explanation: item.explanation.explanation_text,
  }));

const App = (): JSX.Element => {
  const [inputText, setInputText] = useState("");
  const [languageMode, setLanguageMode] = useState<LanguageMode>("auto");
  const [modelType, setModelType] = useState<ModelType>("svm");
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const loadHistoryFromStorage = useAnalysisStore((state) => state.loadHistoryFromStorage);
  const setCurrentAnalysis = useAnalysisStore((state) => state.setCurrentAnalysis);
  const setError = useAnalysisStore((state) => state.setError);

  const { predictText, isLoading, error, result } = useAnalysis();
  const { history, exportCSV } = useHistory();

  useEffect(() => {
    loadHistoryFromStorage();
  }, [loadHistoryFromStorage]);

  const handleSubmit = useCallback(async () => {
    await predictText(inputText, languageMode, modelType);
  }, [inputText, languageMode, modelType, predictText]);

  const handleClear = useCallback(() => {
    setInputText("");
    setError(null);
  }, [setError]);

  const handleSelectAnalysis = useCallback(
    (analysis: AnalysisRecord) => {
      setCurrentAnalysis(analysis);
      setInputText(analysis.text);
      setModelType(analysis.modelType ?? "svm");
      setIsMobileSidebarOpen(false);
    },
    [setCurrentAnalysis]
  );

  const handleCopyResult = useCallback(async () => {
    if (!result) {
      return;
    }
    await copyTextToClipboard(JSON.stringify(result, null, 2));
    toast.success("Analysis copied to clipboard", { duration: 3000 });
  }, [result]);

  const handleExportCurrentJson = useCallback(() => {
    if (!result) {
      return;
    }
    exportToJson("analysis-current.json", result);
    toast.success("Exported successfully", { duration: 3000 });
  }, [result]);

  const handleExportCurrentAndAllCsv = useCallback(() => {
    if (!result) {
      exportCSV();
      return;
    }

    const merged = [result, ...history.filter((item) => item.id !== result.id)];
    exportToCsv("analysis-current-and-history.csv", toCsvRows(merged));
    toast.success("Exported successfully", { duration: 3000 });
  }, [exportCSV, history, result]);

  useKeyboardShortcuts({
    onSubmit: handleSubmit,
    onClear: handleClear,
    onCloseSidebar: () => setIsMobileSidebarOpen(false),
  });

  const hasResult = Boolean(result);

  const { hasSeenWelcome } = useWelcomeScreen();

  const middleContent = useMemo(() => {
    if (isLoading) {
      return (
        <div className="space-y-4">
          <ResultsSkeleton />
          <ExplainerSkeleton />
        </div>
      );
    }

    if (!hasResult || !result) {
      return (
        <article className="rounded-xl border border-white/10 bg-panel p-6 text-sm text-text-muted">
          Submit text to see prediction metrics and explainable AI breakdown.
        </article>
      );
    }

    return (
      <div className="space-y-4">
        <Suspense
          fallback={
            <div className="space-y-4">
              <ResultsSkeleton />
              <ExplainerSkeleton />
            </div>
          }
        >
          <ResultsSection
            analysis={result}
            modelType={result.modelType}
            onCopy={handleCopyResult}
            onExportJson={handleExportCurrentJson}
            onExportCsv={handleExportCurrentAndAllCsv}
          />
          <ExplainerVisualization explanation={result.explanation} />
        </Suspense>
      </div>
    );
  }, [
    handleCopyResult,
    handleExportCurrentAndAllCsv,
    handleExportCurrentJson,
    hasResult,
    isLoading,
    result,
  ]);

  if (!hasSeenWelcome) {
    return <WelcomeScreen onFinish={() => {}} />;
  }

  return (
    <div className="min-h-screen px-4 py-8 md:px-6 lg:px-8 bg-app">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: "var(--color-bg-secondary)",
            color: "var(--color-text-primary)",
            border: "1px solid var(--color-border)",
          },
        }}
      />

      <header className="mb-8 rounded-3xl border border-[var(--color-primary)]/40 bg-[var(--color-bg-secondary)] p-8 shadow-2xl backdrop-blur-md transition-all duration-300 hover:border-[var(--color-primary)]/70 hover:shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold tracking-tight font-display text-[var(--color-text-primary)] md:text-5xl">
              Multilingual Fake News Detection
            </h1>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              React + TypeScript frontend with LIME-based explainability for English and Hindi.
            </p>
          </div>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-sm text-[var(--color-text-primary)] shadow transition duration-300 md:hidden"
            onClick={() => setIsMobileSidebarOpen(true)}
            aria-label="Open history sidebar"
          >
            <FiClock /> History
          </button>
        </div>
      </header>

      <main className="grid gap-6 md:grid-cols-2 lg:grid-cols-3" role="main">
        <section className="space-y-6 md:col-span-2 lg:col-span-1">
          <InputSection
            text={inputText}
            isLoading={isLoading}
            error={error}
            languageMode={languageMode}
            modelType={modelType}
            onTextChange={setInputText}
            onLanguageModeChange={setLanguageMode}
            onModelTypeChange={setModelType}
            onSubmit={handleSubmit}
            onClear={handleClear}
          />

          <article className="pulse-glow rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 text-xs text-[var(--color-text-secondary)] shadow transition-all duration-300 hover:border-[var(--color-primary)]/60">
            Shortcuts: Enter (Windows/Linux textarea submit), Ctrl+Enter (Mac textarea submit), Ctrl/Cmd+L (clear), Esc (close mobile sidebar).
          </article>
        </section>

        <section className="space-y-6 md:col-span-1 lg:col-span-1">{middleContent}</section>

        <section className="md:col-span-1 lg:col-span-1">
          <HistorySidebar
            isMobileOpen={isMobileSidebarOpen}
            onCloseMobile={() => setIsMobileSidebarOpen(false)}
            onSelectAnalysis={handleSelectAnalysis}
          />
        </section>
      </main>

    </div>
  );
};

export default App;
