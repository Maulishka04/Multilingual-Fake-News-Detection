export type LanguageCode = "en" | "hi";
export type LanguageMode = LanguageCode | "auto";
export type ModelType = "svm" | "mbert";

export interface ModelInfo {
  name: string;
  accuracy: string;
  f1Score: string;
  inferenceTime: string;
  supportsLime: boolean;
  description: string;
}

export const MODEL_OPTIONS = [
  { value: "svm" as const, label: "SVM (Fast & Explainable)" },
  { value: "mbert" as const, label: "mBERT (Accurate)" },
] as const;

export const MODEL_INFO: Record<ModelType, ModelInfo> = {
  svm: {
    name: "Linear SVC with TF-IDF",
    accuracy: "~85%",
    f1Score: "~0.81",
    inferenceTime: "~100ms",
    supportsLime: true,
    description: "Fast traditional ML model with LIME explanations",
  },
  mbert: {
    name: "Multilingual BERT",
    accuracy: "91.15%",
    f1Score: "0.8790",
    inferenceTime: "~500ms",
    supportsLime: false,
    description: "Deep learning model with better accuracy",
  },
};

export interface LanguageOption {
  value: LanguageMode;
  label: string;
}

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export const STORAGE_KEYS = {
  history: "mfn_history_v1",
  welcome: "mfn_welcome_v1",
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
