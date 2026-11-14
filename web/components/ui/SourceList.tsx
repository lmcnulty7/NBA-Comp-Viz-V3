import { copy, label } from "@/lib/copy";
import type { Source } from "@/lib/types/contract";

/** Where a value comes from: the files it was read from, and which rows of them were read. */
export function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <details className="sources">
      <summary>{copy.site.sources.title}</summary>
      <ul>
        {sources.map((source) => (
          <li key={`${source.path}|${source.what}|${source.rows}`}>
            <code>{source.path}</code> {source.what} ({label(copy.site.sources.rows, source.rows)})
          </li>
        ))}
      </ul>
    </details>
  );
}
