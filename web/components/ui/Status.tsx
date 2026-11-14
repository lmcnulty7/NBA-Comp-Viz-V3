export type StatusTone = "pass" | "fail" | "warn" | "plain";

/** A status or a verdict: a glyph and a word, so it never rests on colour alone. */
export function Status({ tone, text }: { tone: StatusTone; text: string }) {
  return (
    <span className={`status status-${tone}`}>
      <span className="glyph" aria-hidden="true" />
      {text}
    </span>
  );
}
