import Link from "next/link";

export function NavBar() {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
        <Link href="/" className="font-semibold">
          AI Agents Agency
        </Link>
        <nav className="flex gap-4 text-sm text-neutral-600 dark:text-neutral-400">
          <Link href="/agents" className="hover:text-neutral-900 dark:hover:text-neutral-100">
            Agents
          </Link>
          <Link href="/builder" className="hover:text-neutral-900 dark:hover:text-neutral-100">
            Builder
          </Link>
          <Link href="/settings/integrations" className="hover:text-neutral-900 dark:hover:text-neutral-100">
            Integrations
          </Link>
        </nav>
      </div>
    </header>
  );
}
