import { ANALYSIS_LIMITS, ERROR_MESSAGES, type LanguageCode } from "./constants";

export interface ValidationResult {
  valid: boolean;
  message: string | null;
}

export const validateTextInput = (text: string): ValidationResult => {
  const trimmed = text.trim();

  if (!trimmed) {
    return { valid: false, message: ERROR_MESSAGES.invalidInput };
  }

  if (trimmed.length < ANALYSIS_LIMITS.minChars || trimmed.length > ANALYSIS_LIMITS.maxChars) {
    return { valid: false, message: ERROR_MESSAGES.invalidInput };
  }

  return { valid: true, message: null };
};

export const detectLanguageFromText = (text: string): LanguageCode => {
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  return hasDevanagari ? "hi" : "en";
};
