"use client";

import { useId, useState, type ReactNode } from "react";
import { copy } from "@/lib/copy";
import styles from "./evidence.module.css";

/**
 * The show-the-evidence toggle (PLAN.md Site structure, D25). It wraps a page part; every Evidence block inside
 * stays hidden until the reader presses the button, then each value shows the sources it was read from. The
 * same toggle is used on the possession review and the prep tab.
 */
export function EvidenceScope({ children, intro }: { children: ReactNode; intro?: string }) {
  const [shown, setShown] = useState(false);
  const noteId = useId();
  return (
    <div className={styles.scope} data-evidence={shown ? "on" : "off"}>
      <div className={styles.bar}>
        <button
          type="button"
          className={styles.toggle}
          aria-pressed={shown}
          aria-describedby={noteId}
          onClick={() => setShown((value) => !value)}
          data-testid="evidence-toggle"
        >
          <span className={styles.box} aria-hidden="true" />
          {copy.site.evidence.toggle}
        </button>
        <p id={noteId} className={styles.hint}>
          {intro ?? copy.site.evidence.hint}
        </p>
      </div>
      {children}
    </div>
  );
}
