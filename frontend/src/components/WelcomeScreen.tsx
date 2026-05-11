import { useEffect, useState } from "react";
import useWelcomeScreen from "../hooks/useWelcomeScreen";

const WelcomeScreen = ({ onFinish }: { onFinish?: () => void }) => {
  const { hasSeenWelcome, dismiss } = useWelcomeScreen();
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    if (hasSeenWelcome) {
      setExiting(true);
      const t = setTimeout(() => onFinish?.(), 300);
      return () => clearTimeout(t);
    }
  }, [hasSeenWelcome, onFinish]);

  const handleStart = () => {
    setExiting(true);
    setTimeout(() => {
      dismiss();
      onFinish?.();
    }, 500);
  };

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center welcome-gradient`} style={{ minHeight: "100vh" }}>
      <div className={`mx-4 max-w-3xl rounded-2xl p-8 text-center shadow-2xl ${exiting ? "animate-fade-out" : "animate-fade-up"}`}>
        <h1 className="text-3xl md:text-5xl font-display mb-4" style={{ fontFamily: "Aalto Display, serif", color: "var(--color-text-primary)" }}>
          Fact-Check with Confidence
        </h1>
        <h2 className="text-lg md:text-2xl font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
          Detect Multilingual Fake News Instantly
        </h2>
        <p className="text-sm md:text-base mb-6" style={{ color: "var(--color-text-primary)" }}>
          Our AI-powered system analyzes news in English and Hindi, providing instant predictions with
          AI-powered explanations.
        </p>

        <div className="flex items-center justify-center gap-4">
          <button
            onClick={handleStart}
            className="btn-primary btn-scale-hover rounded-lg px-6 py-3 text-base font-semibold"
            style={{ transition: "all 0.3s ease-in-out" }}
          >
            Let's Get Started
          </button>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
