import { MINUS } from "@/lib/format";
import { showValue, type ShownAs } from "@/lib/tokens";

export type ResultsFile = "scorecard.json" | "results.json";

/**
 * One number of a data file, printed as the file stores it (or moved two decimal places to read as a
 * percentage), never rounded. It names its file and field path, so a browser test can read the same field from
 * the served file and check that the page shows exactly that value.
 */
export function DataValue({
  file,
  field,
  value,
  shown = "as stored",
  signed = false,
}: {
  file: ResultsFile;
  field: string;
  value: number;
  shown?: ShownAs;
  signed?: boolean;
}) {
  return (
    <span className="num" data-file={file} data-field={field} data-shown={shown} data-signed={signed ? "true" : undefined}>
      {signedText(value, shown, signed)}
    </span>
  );
}

/** The text of a value as DataValue prints it. */
export function signedText(value: number, shown: ShownAs, signed: boolean): string {
  const text = showValue(value, shown);
  return signed && value > 0 && !text.startsWith(MINUS) && /[1-9]/.test(text) ? `+${text}` : text;
}

/** One text field of a data file, printed as stored, with its file and field path for the same test. */
export function DataText({ file, field, text, className }: { file: ResultsFile; field: string; text: string; className?: string }) {
  return (
    <span className={className} data-file={file} data-field={field} data-shown="text">
      {text}
    </span>
  );
}
