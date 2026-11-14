import { copy, label } from "@/lib/copy";
import { repoFileUrl } from "@/lib/repo";
import type { Source } from "@/lib/types/contract";

/** A path, linked to its file on GitHub when it is a file of the repository. */
export function RepoPath({ path }: { path: string }) {
  const url = repoFileUrl(path);
  return url ? (
    <a href={url} data-report={path}>
      <code>{path}</code>
    </a>
  ) : (
    <code>{path}</code>
  );
}

/** Where the values of a section come from, each repository file linked to GitHub. */
export function Sources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <details className="sources">
      <summary>{copy.site.sources.title}</summary>
      <ul>
        {sources.map((source) => (
          <li key={`${source.path}|${source.what}|${source.rows}`}>
            <RepoPath path={source.path} /> {source.what} ({label(copy.site.sources.rows, source.rows)})
          </li>
        ))}
      </ul>
    </details>
  );
}
