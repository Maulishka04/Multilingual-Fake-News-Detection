/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontSize: {
        xs: ["0.75rem", { lineHeight: "1rem" }],
        sm: ["0.875rem", { lineHeight: "1.25rem" }],
        base: ["1rem", { lineHeight: "1.5rem" }],
        lg: ["1.125rem", { lineHeight: "1.75rem" }],
        xl: ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.5rem", { lineHeight: "2rem" }],
        "3xl": ["1.875rem", { lineHeight: "2.25rem" }],
        "4xl": ["2.25rem", { lineHeight: "2.5rem" }],
        "5xl": ["3rem", { lineHeight: "1" }],
      },
      spacing: {
        px: "1px",
        0: "0",
        0.5: "0.125rem",
        1: "0.25rem",
        2: "0.5rem",
        3: "0.75rem",
        4: "1rem",
        6: "1.5rem",
        8: "2rem",
        12: "3rem",
        16: "4rem",
      },
      colors: {
        "neon-cyan": "#00D9FF",
        "neon-primary": "#00D9FF",
        "neon-primary-dark": "#0099FF",
        "neon-secondary": "#D700FF",
        "neon-magenta": "#FF00FF",
        "neon-success": "#00FF41",
        "neon-warning": "#FF6B00",
        "neon-danger": "#FF0055",
        background: "#0A0E27",
        surface: "#090C1F",
        panel: "#11162E",
        "text-primary": "#E0E0FF",
        "text-muted": "#A8AED4",
      },
      backgroundImage: {
        "gradient-neon": "linear-gradient(135deg, #0A0E27 0%, #1a1a3e 50%, #0f3460 100%)",
        "gradient-card": "linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(215, 0, 255, 0.1) 100%)",
        "gradient-button": "linear-gradient(90deg, #00D9FF 0%, #0099FF 50%, #D700FF 100%)",
        "gradient-success": "linear-gradient(135deg, #00FF41 0%, #00D9FF 100%)",
        "gradient-danger": "linear-gradient(135deg, #FF0055 0%, #FF6B00 100%)",
      },
      boxShadow: {
        "neon-cyan": "0 0 16px rgba(0, 217, 255, 0.35)",
        "neon-magenta": "0 0 16px rgba(215, 0, 255, 0.35)",
        "neon-green": "0 0 16px rgba(0, 255, 65, 0.25)",
      },
      borderRadius: {
        xl: "12px",
        lg: "10px",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 280ms ease-out",
      },
    },
  },
  plugins: [],
};
