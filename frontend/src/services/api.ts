import axios, { AxiosError } from "axios";
import { BACKEND_URL, ERROR_MESSAGES, type LanguageCode, type ModelType } from "../utils/constants";

export interface ExplanationPayload {
  positive_words: string[];
  negative_words: string[];
  word_scores: Record<string, number>;
  explanation_text: string;
}

export interface PredictionResponse {
  prediction: number;
  confidence: number;
  explanation: ExplanationPayload;
  model?: ModelType;
  inference_time_ms?: number | null;
}

export interface ModelInfo {
  name: ModelType;
  loaded: boolean;
  description: string;
  accuracy: number;
  f1_score: number;
  inference_time_ms: number;
  best_for: string;
  supports_lime: boolean;
}

export interface AvailableModelsResponse {
  models: ModelInfo[];
}

export interface CompareModelResult {
  prediction: number;
  confidence: number;
  inference_time_ms?: number | null;
}

export interface CompareResponse {
  text_preview: string;
  results: Partial<Record<ModelType, CompareModelResult>>;
  errors: Partial<Record<ModelType, string>>;
  agreement: boolean | null;
}

interface PredictPayload {
  text: string;
  language: LanguageCode;
}

const emptyExplanation: ExplanationPayload = {
  positive_words: [],
  negative_words: [],
  word_scores: {},
  explanation_text: "No explanation available",
};

const apiClient = axios.create({
  baseURL: BACKEND_URL,
  timeout: 60_000,
  headers: {
    "Content-Type": "application/json",
  },
});

const getFriendlyErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string }>;

    if (axiosError.code === "ECONNABORTED") {
      return ERROR_MESSAGES.timeout;
    }

    if (!axiosError.response) {
      return ERROR_MESSAGES.network;
    }

    return axiosError.response.data?.detail ?? ERROR_MESSAGES.unknown;
  }

  return ERROR_MESSAGES.unknown;
};

const normalizePredictionResponse = (response: PredictionResponse): PredictionResponse => ({
  prediction: Number(response.prediction),
  confidence: Number(response.confidence),
  model: response.model,
  inference_time_ms: response.inference_time_ms ?? null,
  explanation: {
    positive_words: response.explanation?.positive_words ?? emptyExplanation.positive_words,
    negative_words: response.explanation?.negative_words ?? emptyExplanation.negative_words,
    word_scores: response.explanation?.word_scores ?? emptyExplanation.word_scores,
    explanation_text: response.explanation?.explanation_text ?? emptyExplanation.explanation_text,
  },
});

export const predictText = async (
  text: string,
  language: LanguageCode,
  model: ModelType = "svm"
): Promise<PredictionResponse> => {
  try {
    const payload: PredictPayload = { text, language };
    const response = await apiClient.post<PredictionResponse>(`/predict?model=${model}`, payload);
    return normalizePredictionResponse(response.data);
  } catch (error) {
    throw new Error(getFriendlyErrorMessage(error));
  }
};

export const predictTextWithLime = async (text: string, language: LanguageCode): Promise<PredictionResponse> => {
  try {
    const payload: PredictPayload = { text, language };
    const response = await apiClient.post<PredictionResponse>("/predict-with-lime", payload);
    return normalizePredictionResponse(response.data);
  } catch (error) {
    throw new Error(getFriendlyErrorMessage(error));
  }
};

export const compareModels = async (text: string, language: LanguageCode): Promise<CompareResponse> => {
  try {
    const payload: PredictPayload = { text, language };
    const response = await apiClient.post<CompareResponse>("/compare", payload);
    return response.data;
  } catch (error) {
    throw new Error(getFriendlyErrorMessage(error));
  }
};

export const fetchAvailableModels = async (): Promise<AvailableModelsResponse> => {
  try {
    const response = await apiClient.get<AvailableModelsResponse>("/available-models");
    return response.data;
  } catch (error) {
    throw new Error(getFriendlyErrorMessage(error));
  }
};

export const fetchModelInfo = async (modelName: ModelType): Promise<ModelInfo> => {
  try {
    const response = await apiClient.get<ModelInfo>(`/model-info/${modelName}`);
    return response.data;
  } catch (error) {
    throw new Error(getFriendlyErrorMessage(error));
  }
};
