export const ResultsSkeleton = (): JSX.Element => (
  <section className="animate-pulse rounded-xl border border-neon-primary/30 bg-panel p-5 shadow-neon-cyan">
    <div className="mb-4 h-6 w-40 rounded bg-white/10" />
    <div className="grid gap-4 sm:grid-cols-2">
      <div className="h-24 rounded-lg bg-white/10" />
      <div className="h-24 rounded-lg bg-white/10" />
    </div>
    <div className="mt-4 h-16 rounded-lg bg-white/10" />
  </section>
);

export const ExplainerSkeleton = (): JSX.Element => (
  <section className="animate-pulse rounded-xl border border-neon-secondary/30 bg-panel p-5 shadow-neon-magenta">
    <div className="mb-4 h-6 w-56 rounded bg-white/10" />
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="h-20 rounded-lg bg-white/10" />
      <div className="h-20 rounded-lg bg-white/10" />
    </div>
    <div className="mt-4 h-48 rounded-lg bg-white/10" />
  </section>
);
