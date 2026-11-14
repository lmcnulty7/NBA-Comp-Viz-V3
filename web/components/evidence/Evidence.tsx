import { copy, fill, label } from "@/lib/copy";
import type { Source } from "@/lib/types/contract";
import styles from "./evidence.module.css";

/**
 * The evidence of one value: the files it was read from and which rows of them. Shown only while the
 * EvidenceScope around it is switched on. With no source, it says so instead of showing nothing.
 */
export function Evidence({ what, sources, testId }: { what: string; sources: readonly Source[]; testId?: string }) {
  return (
    <div className={styles.block} data-testid={testId ?? "evidence"} data-evidence-for={what}>
      <p className={styles.title}>{fill(copy.site.evidence.for, { what })}</p>
      {sources.length === 0 ? (
        <p className={styles.empty}>{copy.site.evidence.none}</p>
      ) : (
        <ul className={styles.list}>
          {sources.map((source) => (
            <li key={`${source.path}|${source.what}|${source.rows}`}>
              <code>{source.path}</code> <span>{source.what}</span>{" "}
              <span className={styles.rows} data-rows={source.rows}>
                {label(copy.site.sources.rows, source.rows)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
