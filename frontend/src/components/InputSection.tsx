import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { FiRefreshCw, FiSend, FiX } from "react-icons/fi";
import { ANALYSIS_LIMITS, LANGUAGE_OPTIONS, type LanguageMode, type ModelType } from "../utils/constants";

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

  const [localText, setLocalText] = useState<string>(text ?? "");
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    // sync when parent text prop changes (e.g., selecting history)
    setLocalText(text ?? "");
  }, [text]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const characterCount = useMemo(() => localText.length, [localText]);

  const handleLocalChange = useCallback(
    (value: string) => {
      setLocalText(value);
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
      // debounce 300ms before propagating to parent
      timerRef.current = window.setTimeout(() => {
        onTextChange(value);
        timerRef.current = null;
      }, 300);
    },
    [onTextChange]
  );

  return (
    <section className="rounded-2xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-6 shadow-lg transition-all duration-300 hover:border-[var(--color-primary)]/60">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold font-display text-[var(--color-text-primary)]">Analyze News</h2>
        <div className="text-sm text-[var(--color-text-secondary)]" aria-live="polite">
          {characterCount}/{ANALYSIS_LIMITS.maxChars}
        </div>
      </header>

      <div className="mb-4 grid gap-3 md:grid-cols-3">
        <label className="flex flex-col gap-1 text-sm text-[var(--color-text-secondary)]" htmlFor="language-mode">
          Language
          <select
            id="language-mode"
            aria-label="Select language mode"
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-[var(--color-text-primary)] focus:border-[var(--color-primary)] focus:outline-none transition duration-300"
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

        <label className="flex flex-col gap-1 text-sm text-[var(--color-text-secondary)]" htmlFor="model-type">
          Model
          <select
            id="model-type"
            aria-label="Select prediction model"
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-[var(--color-text-primary)] focus:border-[var(--color-primary)] focus:outline-none transition duration-300"
            value={modelType}
            onChange={(event) => onModelTypeChange(event.target.value as ModelType)}
            disabled={isLoading}
          >
            <option value="svm">SVM (Fast & Explainable)</option>
            <option value="mbert">mBERT (Accurate)</option>
          </select>
        </label>
      </div>

      <label className="sr-only" htmlFor="news-input">
        News content
      </label>
      <textarea
        id="news-input"
        ref={textareaRef}
        aria-label="News content input"
        className="h-44 w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-4 py-3 text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/60 focus:border-[var(--color-primary)] focus:outline-none transition duration-300"
        placeholder="Paste or type news content here..."
        value={localText}
        onChange={(event) => handleLocalChange(event.target.value)}
        maxLength={ANALYSIS_LIMITS.maxChars}
      />

      {error ? <p className="mt-3 text-sm text-[var(--color-danger)]">{error}</p> : null}

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-primary)]/60 bg-[var(--color-primary)] px-5 py-2.5 text-sm font-medium text-white transition duration-300 hover:bg-[#007A75] hover:shadow-lg disabled:opacity-60 disabled:cursor-not-allowed"
          aria-label="Submit text for analysis"
        >
          {isLoading ? <FiRefreshCw className="animate-spin" /> : <FiSend />}
          {isLoading ? "Processing..." : "Analyze"}
        </button>

        <button
          type="button"
          onClick={onClear}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)]/50 px-5 py-2.5 text-sm font-medium text-[var(--color-text-primary)] transition duration-300 hover:bg-[var(--color-bg-secondary)] disabled:opacity-60 disabled:cursor-not-allowed"
          aria-label="Clear input"
        >
          <FiX />
          Clear
        </button>
      </div>
    </section>
  );
};
