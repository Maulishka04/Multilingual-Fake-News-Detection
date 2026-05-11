import { ErrorBoundary } from "react-error-boundary";
import type { FallbackProps } from "react-error-boundary";

const ErrorFallback = ({ error, resetErrorBoundary }: FallbackProps): JSX.Element => (
  <div className="mx-auto mt-10 w-full max-w-xl rounded-xl border border-[var(--color-danger)]/50 bg-[var(--color-bg-secondary)] p-6 text-center">
    <h1 className="text-2xl font-semibold text-[var(--color-danger)]">Something went wrong</h1>
    <p className="mt-2 text-sm text-[var(--color-text-secondary)]">
      {error instanceof Error ? error.message : "Unexpected runtime error"}
    </p>
    <button
      type="button"
      onClick={resetErrorBoundary}
      className="mt-4 rounded-lg border border-[var(--color-primary)]/60 px-4 py-2 text-[var(--color-primary)] transition duration-300 hover:bg-[var(--color-primary)]/10"
    >
      Try again
    </button>
  </div>
);

interface AppErrorBoundaryProps {
  children: JSX.Element;
}

export const AppErrorBoundary = ({ children }: AppErrorBoundaryProps): JSX.Element => (
  <ErrorBoundary FallbackComponent={ErrorFallback}>{children}</ErrorBoundary>
);
