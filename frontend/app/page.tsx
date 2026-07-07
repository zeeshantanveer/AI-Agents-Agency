import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center gap-4 py-24 text-center">
      <h1 className="text-3xl font-semibold">AI Agents Agency</h1>
      <p className="max-w-md text-neutral-500">
        Production-ready built-in AI agents, and a prompt-to-agent generator.
      </p>
      <div className="mt-4 flex gap-3">
        <Link
          href="/agents"
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
        >
          Browse agents
        </Link>
        <Link
          href="/builder"
          className="rounded-md border border-neutral-300 px-4 py-2 text-sm font-medium dark:border-neutral-700"
        >
          Build an agent
        </Link>
      </div>
    </div>
  );
}
