import type { Ctx2D } from "@/components/court/draw";

export interface Call {
  name: string;
  args: unknown[];
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
  dash: number[];
  font: string;
}

/** A stand-in for a canvas context that records every call with the drawing state at that moment. */
export class RecordingContext implements Ctx2D {
  fillStyle: string | CanvasGradient | CanvasPattern = "";
  strokeStyle: string | CanvasGradient | CanvasPattern = "";
  lineWidth = 1;
  lineCap: CanvasLineCap = "butt";
  lineJoin: CanvasLineJoin = "miter";
  font = "";
  textAlign: CanvasTextAlign = "start";
  textBaseline: CanvasTextBaseline = "alphabetic";
  readonly calls: Call[] = [];
  private dash: number[] = [];
  private stack: { fillStyle: string; strokeStyle: string; lineWidth: number; dash: number[] }[] = [];

  private record(name: string, ...args: unknown[]): void {
    this.calls.push({
      name,
      args,
      fillStyle: String(this.fillStyle),
      strokeStyle: String(this.strokeStyle),
      lineWidth: this.lineWidth,
      dash: [...this.dash],
      font: this.font,
    });
  }

  named(name: string): Call[] {
    return this.calls.filter((call) => call.name === name);
  }

  save(): void {
    this.stack.push({
      fillStyle: String(this.fillStyle),
      strokeStyle: String(this.strokeStyle),
      lineWidth: this.lineWidth,
      dash: [...this.dash],
    });
    this.record("save");
  }

  restore(): void {
    const state = this.stack.pop();
    if (state) {
      this.fillStyle = state.fillStyle;
      this.strokeStyle = state.strokeStyle;
      this.lineWidth = state.lineWidth;
      this.dash = state.dash;
    }
    this.record("restore");
  }

  setTransform(...args: number[]): void {
    this.record("setTransform", ...args);
  }

  clearRect(...args: number[]): void {
    this.record("clearRect", ...args);
  }

  fillRect(...args: number[]): void {
    this.record("fillRect", ...args);
  }

  beginPath(): void {
    this.record("beginPath");
  }

  closePath(): void {
    this.record("closePath");
  }

  rect(...args: number[]): void {
    this.record("rect", ...args);
  }

  clip(): void {
    this.record("clip");
  }

  moveTo(...args: number[]): void {
    this.record("moveTo", ...args);
  }

  lineTo(...args: number[]): void {
    this.record("lineTo", ...args);
  }

  arc(...args: unknown[]): void {
    this.record("arc", ...args);
  }

  fill(): void {
    this.record("fill");
  }

  stroke(): void {
    this.record("stroke");
  }

  setLineDash(segments: number[]): void {
    this.dash = [...segments];
    this.record("setLineDash", segments);
  }

  fillText(...args: unknown[]): void {
    this.record("fillText", ...args);
  }
}

export const TEST_PALETTE = {
  ground: "ground",
  line: "line",
  offense: "offense",
  offenseOn: "offense-on",
  defense: "defense",
  defenseOn: "defense-on",
  neutral: "neutral",
  neutralOn: "neutral-on",
  ghost: "ghost",
  truth: "truth",
  ball: "ball",
  marker: "marker",
  fontFamily: "Test Sans",
};
