import { useCallback } from "react";
import { toast } from "react-hot-toast";
import { predictText as predictTextSimple, predictTextWithLime } from "../services/api";
import { useAnalysisStore, type AnalysisRecord } from "../store/analysisStore";
import { type LanguageCode, type LanguageMode, type ModelType } from "../utils/constants";
import { detectLanguageFromText, validateTextInput } from "../utils/validators";

const createAnalysisRecord = (
  text: string,
  language: LanguageCode,
  response: Pick<AnalysisRecord, "prediction" | "confidence" | "explanation" | "model" | "inference_time_ms">
): AnalysisRecord => ({
  id: crypto.randomUUID(),
  text,
  language,
  prediction: response.prediction,
  confidence: response.confidence,
  explanation: response.explanation,
  model: response.model,
  inference_time_ms: response.inference_time_ms ?? null,
  createdAt: new Date().toISOString(),
});

export const useAnalysis = () => {
  const {
    isLoading,
    error,
    currentAnalysis,
    setLoading,
    setError,
    addToHistory,
    setSelectedLanguage,
  } = useAnalysisStore();

  const predictText = useCallback(
    async (text: string, languageMode: LanguageMode, modelType: ModelType = "svm"): Promise<AnalysisRecord | null> => {
      const validation = validateTextInput(text);
      if (!validation.valid) {
        setError(validation.message);
        toast.error(validation.message ?? "Invalid input");
        return null;
      }

      const language: LanguageCode = languageMode === "auto" ? detectLanguageFromText(text) : languageMode;

      setLoading(true);
      setError(null);

      try {
        let response;
        if (modelType === "svm") {
          // SVM supports LIME; try LIME first, fall back to plain predict
          try {
            response = await predictTextWithLime(text, language);
          } catch {
            response = await predictTextSimple(text, language, "svm");
          }
        } else {
          // mBERT: use plain predict endpoint
          response = await predictTextSimple(text, language, modelType);
        }

        const analysisRecord = createAnalysisRecord(text, language, {
          ...response,
          model: modelType,
          inference_time_ms: response.inference_time_ms ?? null,
        });
        setSelectedLanguage(language);
        addToHistory(analysisRecord);
        toast.success("Analysis saved to history", { duration: 3000 });

        return analysisRecord;
      } catch (apiError) {
        const errorMessage = apiError instanceof Error ? apiError.message : "Unexpected error";
        setError(errorMessage);
        toast.error(errorMessage, { duration: 3000 });
        return null;
      } finally {
        setLoading(false);
      }
    },
    [addToHistory, setError, setLoading, setSelectedLanguage]
  );

  return {
    predictText,
    isLoading,
    error,
    result: currentAnalysis,
  };
};
