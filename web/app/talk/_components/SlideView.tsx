"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useSyncExternalStore, type ReactNode } from "react";
import deck from "@/content/talk/deck.json";
import { fill } from "@/lib/copy";
import type { Slide } from "@/lib/talk";
import styles from "./talk.module.css";

const NOTES_KEY = "talk-notes";

function slideHref(id: string): string {
  return `/talk/${id}/`;
}

const NOTES_EVENT = "talk-notes-change";

function readNotesChoice(): boolean {
  try {
    return window.sessionStorage.getItem(NOTES_KEY) === "on";
  } catch {
    return false;
  }
}

function saveNotesChoice(on: boolean): void {
  try {
    window.sessionStorage.setItem(NOTES_KEY, on ? "on" : "off");
  } catch {
    // Storage can be blocked; the notes then follow the choice for this page only.
    notesFallback = on;
  }
  window.dispatchEvent(new Event(NOTES_EVENT));
}

let notesFallback = false;

function subscribeNotes(onChange: () => void): () => void {
  window.addEventListener(NOTES_EVENT, onChange);
  return () => window.removeEventListener(NOTES_EVENT, onChange);
}

function notesSnapshot(): boolean {
  return readNotesChoice() || notesFallback;
}

/** Keys that belong to a control the focus is on (a time bar, a button), so the deck leaves them alone. */
function ownedByControl(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.closest("input, select, textarea, [contenteditable='true'], video") !== null;
}

/**
 * One slide of the talk with the keys of a slide deck: the arrow keys, Page Up and Page Down, Home and End move
 * between slides, N shows or hides the presenter notes. The notes stay on from slide to slide.
 */
export function SlideView({
  slide,
  total,
  previous,
  next,
  first,
  last,
  children,
}: {
  slide: Slide;
  total: number;
  previous: string | null;
  next: string | null;
  first: string;
  last: string;
  children?: ReactNode;
}) {
  const router = useRouter();
  const notes = useSyncExternalStore(subscribeNotes, notesSnapshot, () => false);

  const toggleNotes = useCallback(() => saveNotesChoice(!notesSnapshot()), []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.defaultPrevented) return;
      if (ownedByControl(event.target)) return;
      let target: string | null = null;
      if (event.key === "ArrowRight" || event.key === "PageDown") target = next;
      else if (event.key === "ArrowLeft" || event.key === "PageUp") target = previous;
      else if (event.key === "Home") target = first;
      else if (event.key === "End") target = last;
      else if (event.key === "n" || event.key === "N") {
        event.preventDefault();
        toggleNotes();
        return;
      } else return;
      event.preventDefault();
      if (target && target !== slide.id) router.push(slideHref(target));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, next, previous, first, last, slide.id, toggleNotes]);

  useEffect(() => {
    if (next) router.prefetch(slideHref(next));
    if (previous) router.prefetch(slideHref(previous));
  }, [router, next, previous]);

  const ui = deck.ui;
  return (
    <div className={styles.deck} data-notes={notes ? "on" : "off"} data-slide={slide.id}>
      <article className={styles.slide} aria-labelledby="slide-title">
        <p className={styles.count} data-testid="slide-count">
          {fill(ui.count, { n: slide.number, total })}
        </p>
        <div className={slide.visual === "none" ? styles.textOnly : styles.body}>
          <div className={styles.copy}>
            <h1 id="slide-title" className={slide.scene === "headline" ? styles.headline : styles.title}>
              {slide.title}
            </h1>
            {slide.blocks.map((block, index) =>
              block.kind === "p" ? (
                <p key={`p${index}`}>{block.text}</p>
              ) : (
                <ul key={`l${index}`} className={styles.list}>
                  {block.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ),
            )}
          </div>
          {children ? <div className={styles.visual}>{children}</div> : null}
        </div>
      </article>

      <nav className={styles.controls} aria-label={ui.controls}>
        {previous ? (
          <Link href={slideHref(previous)} rel="prev" className={styles.navLink}>
            {ui.previous}
          </Link>
        ) : (
          <span />
        )}
        <Link href="/talk/" className={styles.navLink}>
          {ui.index}
        </Link>
        <button type="button" className={styles.notesButton} onClick={toggleNotes} aria-pressed={notes} aria-controls="notes">
          {notes ? ui.hideNotes : ui.showNotes}
        </button>
        {next ? (
          <Link href={slideHref(next)} rel="next" className={styles.navLink}>
            {ui.next}
          </Link>
        ) : (
          <span />
        )}
      </nav>
      <p className={styles.keys}>{ui.keys}</p>

      <aside id="notes" className={styles.notes} hidden={!notes} aria-label={ui.notes} data-testid="presenter-notes">
        <h2 className={styles.notesTitle}>{ui.notes}</h2>
        {slide.notes.length > 0 ? (
          <ul>
            {slide.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        ) : (
          <p>{ui.noNotes}</p>
        )}
      </aside>
    </div>
  );
}
