export type LanguageCode = "en" | "hi";
export type LanguageMode = LanguageCode | "auto";
export type ModelType = "svm" | "mbert";

export interface LanguageOption {
  value: LanguageMode;
  label: string;
}

export interface ModelOption {
  value: ModelType;
  label: string;
  description: string;
}

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export const STORAGE_KEYS = {
  history: "mfn_history_v1",
} as const;

export const ANALYSIS_LIMITS = {
  minChars: 10,
  maxChars: 5000,
  maxHistoryItems: 50,
} as const;

export const LANGUAGE_OPTIONS: ReadonlyArray<LanguageOption> = [
  { value: "auto", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi" },
];

export const MODEL_OPTIONS: ReadonlyArray<ModelOption> = [
  { value: "svm", label: "SVM (TF-IDF)", description: "Fast · Explainable · ~85% accuracy" },
  { value: "mbert", label: "mBERT", description: "Accurate · Multilingual · 91.15% accuracy" },
];

export const ERROR_MESSAGES = {
  network: "Unable to connect to backend",
  timeout: "Request took too long, try again",
  invalidInput: "Please enter text (min 10 chars, max 5000)",
  unknown: "Something went wrong. Please try again.",
} as const;

export const KEYBOARD_SHORTCUTS = {
  submitWindowsLinux: "Enter",
  submitMac: "Ctrl+Enter",
  clear: "Ctrl/Cmd+L",
  closeSidebar: "Esc",
} as const;
