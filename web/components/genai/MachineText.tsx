import { Evidence } from "@/components/evidence/Evidence";
import { copy, fill } from "@/lib/copy";
import type { GeneratedText, GeneratedTexts } from "@/lib/types/contract";
import styles from "./machine-text.module.css";

/**
 * A machine-written text (PLAN.md P3), labelled as one, with the model and the prompt version that wrote it. The
 * page passes only a text that passed its own number check (lib/genai.ts, checkedText); with none, nothing shows.
 * The evidence names the generator and the prompt; the numbers themselves keep their own evidence on the page.
 */
export function MachineText({
  item,
  texts,
  className,
}: {
  item: GeneratedText | null;
  texts: GeneratedTexts | null;
  className?: string;
}) {
  if (!item || !texts) return null;
  const text = copy.site.machineText;
  const own = texts.sources.filter((source) => source.path.startsWith("export/"));
  return (
    <section
      className={[className, "section", styles.block].filter(Boolean).join(" ")}
      aria-labelledby="machine-text"
      data-testid="machine-text"
      data-model={item.model}
      data-prompt={item.prompt_version}
    >
      <h2 id="machine-text">{text.title}</h2>
      <p className={styles.label}>{text.label}</p>
      <p className={styles.body}>{item.text}</p>
      <p className="small muted">{fill(text.note, { model: item.model, version: item.prompt_version })}</p>
      {item.model === "fixture" ? <p className="small muted">{text.fixture}</p> : null}
      <Evidence what={text.title} sources={own.length > 0 ? own : texts.sources} />
    </section>
  );
}
