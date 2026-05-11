import { useCallback } from "react";
import { useAnalysisStore } from "../store/analysisStore";

export const useWelcomeScreen = () => {
  const hasSeenWelcome = useAnalysisStore((s) => s.hasSeenWelcome);
  const setHasSeenWelcome = useAnalysisStore((s) => s.setHasSeenWelcome);

  const dismiss = useCallback(() => {
    setHasSeenWelcome(true);
  }, [setHasSeenWelcome]);

  return { hasSeenWelcome, dismiss } as const;
};

export default useWelcomeScreen;
