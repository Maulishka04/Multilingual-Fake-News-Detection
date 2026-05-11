export const ResultsSkeleton = (): JSX.Element => (
  <section className="animate-pulse rounded-xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-5 shadow transition duration-300">
    <div className="mb-4 h-6 w-40 rounded bg-[var(--color-bg-primary)]/60" />
    <div className="grid gap-4 sm:grid-cols-2">
      <div className="h-24 rounded-lg bg-[var(--color-bg-primary)]/60" />
      <div className="h-24 rounded-lg bg-[var(--color-bg-primary)]/60" />
    </div>
    <div className="mt-4 h-16 rounded-lg bg-[var(--color-bg-primary)]/60" />
  </section>
);

export const ExplainerSkeleton = (): JSX.Element => (
  <section className="animate-pulse rounded-xl border border-[var(--color-primary)]/30 bg-[var(--color-bg-secondary)] p-5 shadow transition duration-300">
    <div className="mb-4 h-6 w-56 rounded bg-[var(--color-bg-primary)]/60" />
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="h-20 rounded-lg bg-[var(--color-bg-primary)]/60" />
      <div className="h-20 rounded-lg bg-[var(--color-bg-primary)]/60" />
    </div>
    <div className="mt-4 h-48 rounded-lg bg-[var(--color-bg-primary)]/60" />
  </section>
);
