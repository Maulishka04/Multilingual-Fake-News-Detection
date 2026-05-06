import { create } from "zustand";
import { ANALYSIS_LIMITS, STORAGE_KEYS, type LanguageCode } from "../utils/constants";
import type { PredictionResponse } from "../services/api";

export interface AnalysisRecord extends PredictionResponse {
  id: string;
  text: string;
  language: LanguageCode;
  createdAt: string;
}

interface AnalysisStoreState {
  currentAnalysis: AnalysisRecord | null;
  history: AnalysisRecord[];
  isLoading: boolean;
  error: string | null;
  selectedLanguage: LanguageCode;
  setCurrentAnalysis: (analysis: AnalysisRecord | null) => void;
  addToHistory: (analysis: AnalysisRecord) => void;
  clearHistory: () => void;
  removeFromHistory: (id: string) => void;
  loadHistoryFromStorage: () => void;
  setError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
  setSelectedLanguage: (language: LanguageCode) => void;
}

const saveHistory = (history: AnalysisRecord[]): void => {
  if (typeof window === "undefined") {
    return;
  }
  localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(history));
};

const getHistoryFromStorage = (): AnalysisRecord[] => {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const serialized = localStorage.getItem(STORAGE_KEYS.history);
    if (!serialized) {
      return [];
    }

    const parsed: unknown = JSON.parse(serialized);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed as AnalysisRecord[];
  } catch {
    return [];
  }
};

export const useAnalysisStore = create<AnalysisStoreState>((set) => ({
  currentAnalysis: null,
  history: [],
  isLoading: false,
  error: null,
  selectedLanguage: "en",

  setCurrentAnalysis: (analysis) => {
    set({ currentAnalysis: analysis });
  },

  addToHistory: (analysis) => {
    set((state) => {
      const nextHistory = [analysis, ...state.history].slice(0, ANALYSIS_LIMITS.maxHistoryItems);
      saveHistory(nextHistory);
      return {
        history: nextHistory,
        currentAnalysis: analysis,
      };
    });
  },

  clearHistory: () => {
    saveHistory([]);
    set({ history: [] });
  },

  removeFromHistory: (id) => {
    set((state) => {
      const nextHistory = state.history.filter((item) => item.id !== id);
      saveHistory(nextHistory);
      return {
        history: nextHistory,
      };
    });
  },

  loadHistoryFromStorage: () => {
    const loadedHistory = getHistoryFromStorage();
    set({ history: loadedHistory });
  },

  setError: (error) => {
    set({ error });
  },

  setLoading: (loading) => {
    set({ isLoading: loading });
  },

  setSelectedLanguage: (language) => {
    set({ selectedLanguage: language });
  },
}));
