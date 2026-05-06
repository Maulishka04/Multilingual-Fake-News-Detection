# Multilingual Fake News Detection Frontend

React 18 + TypeScript + Vite frontend for fake news detection with Explainable AI (LIME) visualization.

## Features

- Dark neon UI with responsive layout
- Text analysis in English and Hindi, with auto-detect mode
- Prediction results with confidence gauge and probability breakdown
- LIME explanation chips and top-word importance chart (Recharts)
- History sidebar with search/filter/delete/clear
- LocalStorage persistence (max 50 analyses)
- Export options: JSON/CSV and clipboard copy
- Keyboard shortcuts:
  - Enter in textarea (Windows/Linux) or Ctrl+Enter (Mac): submit
  - Ctrl/Cmd+L: clear input
  - Esc: close mobile history drawer

## Backend Configuration

Backend URL is read from `VITE_BACKEND_URL`.

1. Copy env template:

```bash
cp .env.example .env.local
```

2. Set backend URL in `.env.local`:

```bash
VITE_BACKEND_URL=http://localhost:8000
```

## Setup

```bash
npm install
npm run dev
```

## Scripts

- `npm run dev` - Start local development server
- `npm run build` - TypeScript build + Vite production build
- `npm run preview` - Preview production build locally
- `npm run type-check` - Run strict TypeScript checks

## Folder Structure

```text
frontend/
  src/
    components/
    hooks/
    services/
    store/
    styles/
    utils/
```

## Theme Customization

Update neon palette, shadows, and animations in `tailwind.config.js`.

Global background and style layers are defined in:

- `src/styles/globals.css`
- `src/styles/neon.css`
