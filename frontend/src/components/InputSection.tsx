import { useEffect, useMemo, useRef } from "react";
import { FiRefreshCw, FiSend, FiX } from "react-icons/fi";
import { ANALYSIS_LIMITS, LANGUAGE_OPTIONS, MODEL_OPTIONS, type LanguageMode, type ModelType } from "../utils/constants";

interface InputSectionProps {
  text: string;
  isLoading: boolean;
  error: string | null;
  languageMode: LanguageMode;
  modelType: ModelType;
  onTextChange: (value: string) => void;
  onLanguageModeChange: (value: LanguageMode) => void;
  onModelTypeChange: (value: ModelType) => void;
  onSubmit: () => void;
  onClear: () => void;
}

export const InputSection = ({
  text,
  isLoading,
  error,
  languageMode,
  modelType,
  onTextChange,
  onLanguageModeChange,
  onModelTypeChange,
  onSubmit,
  onClear,
}: InputSectionProps): JSX.Element => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const characterCount = useMemo(() => text.length, [text]);

  const selectedModelOption = MODEL_OPTIONS.find((o) => o.value === modelType);

  return (
    <section className="glow-cyan rounded-2xl border border-neon-cyan/30 bg-gradient-card p-6 shadow-2xl backdrop-blur-md transition-all duration-300 hover:border-neon-cyan/60">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-text-primary">Analyze News</h2>
        <div className="text-sm text-text-muted" aria-live="polite">
          {characterCount}/{ANALYSIS_LIMITS.maxChars}
        </div>
      </header>

      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-text-muted" htmlFor="language-mode">
          Language
          <select
            id="language-mode"
            aria-label="Select language mode"
            className="rounded-lg border border-white/20 bg-surface px-3 py-2 text-text-primary focus:border-neon-primary focus:outline-none"
            value={languageMode}
            onChange={(event) => onLanguageModeChange(event.target.value as LanguageMode)}
            disabled={isLoading}
          >
            {LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm text-text-muted" htmlFor="model-type">
          Model
          <select
            id="model-type"
            aria-label="Select AI model"
            className="rounded-lg border border-white/20 bg-surface px-3 py-2 text-text-primary focus:border-neon-primary focus:outline-none"
            value={modelType}
            onChange={(event) => onModelTypeChange(event.target.value as ModelType)}
            disabled={isLoading}
          >
            {MODEL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {selectedModelOption ? (
        <p className="mb-3 text-xs text-text-muted">{selectedModelOption.description}</p>
      ) : null}

      <label className="sr-only" htmlFor="news-input">
        News content
      </label>
      <textarea
        id="news-input"
        ref={textareaRef}
        aria-label="News content input"
        className="h-44 w-full rounded-xl border border-white/20 bg-surface px-4 py-3 text-text-primary placeholder:text-text-muted focus:border-neon-primary focus:outline-none"
        placeholder="Paste or type news content here..."
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        maxLength={ANALYSIS_LIMITS.maxChars}
      />

      {error ? <p className="mt-3 text-sm text-neon-danger">{error}</p> : null}

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-lg border border-neon-primary/60 bg-gradient-button px-5 py-2.5 text-sm font-medium text-text-primary transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
          aria-label="Submit text for analysis"
        >
          {isLoading ? <FiRefreshCw className="animate-spin" /> : <FiSend />}
          {isLoading ? "Processing..." : "Analyze"}
        </button>

        <button
          type="button"
          onClick={onClear}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-lg border border-white/25 bg-white/5 px-5 py-2.5 text-sm font-medium text-text-primary transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
          aria-label="Clear input"
        >
          <FiX />
          Clear
        </button>
      </div>
    </section>
  );
};
