import { useMemo } from "react";
import { useAnalysisStore, type AnalysisRecord } from "../store/analysisStore";
import { exportToCsv, exportToJson } from "../utils/exporters";

const toCsvRows = (history: AnalysisRecord[]): Array<Record<string, string | number>> =>
  history.map((item) => ({
    id: item.id,
    text: item.text,
    language: item.language,
    prediction: item.prediction,
    confidence: Number((item.confidence * 100).toFixed(2)),
    createdAt: item.createdAt,
    explanation: item.explanation.explanation_text,
  }));

export const useHistory = () => {
  const history = useAnalysisStore((state) => state.history);
  const addAnalysis = useAnalysisStore((state) => state.addToHistory);
  const deleteAnalysis = useAnalysisStore((state) => state.removeFromHistory);
  const clearHistory = useAnalysisStore((state) => state.clearHistory);

  const sortedHistory = useMemo(
    () => [...history].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt)),
    [history]
  );

  const exportJSON = (): void => {
    exportToJson("analysis-history.json", sortedHistory);
  };

  const exportCSV = (): void => {
    exportToCsv("analysis-history.csv", toCsvRows(sortedHistory));
  };

  return {
    history: sortedHistory,
    addAnalysis,
    deleteAnalysis,
    clearHistory,
    exportJSON,
    exportCSV,
  };
};
