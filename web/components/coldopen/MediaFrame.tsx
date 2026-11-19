"use client";

import { useEffect, useRef, useState } from "react";
import { copy } from "@/lib/copy";
import type { DemoBox } from "@/lib/types/contract";
import styles from "./coldopen.module.css";

export interface StoryMedia {
  /** The clip URL: the private build only. */
  clip: string | null;
  /** The still URL, when the still is part of this build. */
  still: string | null;
  /** The same still as a WebP, when this build holds one: shown where the browser takes WebP, the JPEG elsewhere. */
  stillWebp?: string | null;
  width: number;
  height: number;
}

export interface MediaFrameProps {
  media: StoryMedia;
  /** Boxes to draw over the picture, in clip pixels, or null for none. */
  boxes: DemoBox[] | null;
  /** Draw a box that matched no sensor player as an outline only. */
  outlineUnmatched?: boolean;
  /** Show the clip (private build) rather than the still. */
  moving?: boolean;
  /** Play the clip now. */
  play?: boolean;
  /** Called with the clip time while the clip plays. */
  onTime?: (t: number) => void;
  /** Called once when the clip cannot be played and the still is shown instead. */
  onFailed?: () => void;
  dim?: boolean;
  className?: string;
}

/**
 * The broadcast picture of the story: the clip on the private build, the still on the public one, with the
 * tracker's boxes drawn over it. The boxes sit in an SVG with the clip's own pixel size as its view box, so they
 * stay on the players at every width. A clip that cannot be played falls back to the still.
 */
export function MediaFrame({ media, boxes, outlineUnmatched = false, moving = false, play = false, onTime, onFailed, dim = false, className }: MediaFrameProps) {
  const video = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  const showClip = moving && media.clip !== null && !failed;

  useEffect(() => {
    const element = video.current;
    if (!element || !showClip) return;
    if (play) element.play().catch(() => undefined);
    else element.pause();
  }, [play, showClip]);

  useEffect(() => {
    const element = video.current;
    if (!element || !showClip || !onTime) return;
    let frame = 0;
    const tick = () => {
      onTime(element.currentTime);
      frame = requestAnimationFrame(tick);
    };
    const start = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(tick);
    };
    const stop = () => cancelAnimationFrame(frame);
    const seeked = () => onTime(element.currentTime);
    element.addEventListener("play", start);
    element.addEventListener("pause", stop);
    element.addEventListener("seeked", seeked);
    if (!element.paused) start();
    return () => {
      stop();
      element.removeEventListener("play", start);
      element.removeEventListener("pause", stop);
      element.removeEventListener("seeked", seeked);
    };
  }, [showClip, onTime]);

  const classes = [styles.media, dim ? styles.dim : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <div className={classes} style={{ aspectRatio: `${media.width} / ${media.height}` }} data-media={showClip ? "clip" : media.still ? "still" : "none"}>
      {showClip ? (
        <video
          ref={video}
          className={styles.picture}
          src={media.clip ?? undefined}
          poster={media.stillWebp ?? media.still ?? undefined}
          muted
          loop
          playsInline
          preload="metadata"
          aria-label={copy.home.ui.clipLabel}
          onError={() => {
            setFailed(true);
            onFailed?.();
          }}
        />
      ) : media.still ? (
        // A plain picture: the still is a static file of the export (a WebP where the browser takes it, the JPEG
        // elsewhere), and next/image would add nothing here.
        <picture>
          {media.stillWebp ? <source srcSet={media.stillWebp} type="image/webp" /> : null}
          <img className={styles.picture} src={media.still} alt={copy.home.ui.stillAlt} width={media.width} height={media.height} />
        </picture>
      ) : (
        <p className={styles.noPicture}>{copy.home.ui.noStill}</p>
      )}
      {boxes && boxes.length > 0 ? (
        <svg
          className={styles.boxes}
          viewBox={`0 0 ${media.width} ${media.height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={copy.home.ui.boxesLabel}
        >
          {boxes.map((box) => (
            <rect
              key={`${box.track}|${box.x}|${box.y}`}
              x={box.x}
              y={box.y}
              width={box.w}
              height={box.h}
              className={box.status === "unmatched" && outlineUnmatched ? styles.boxOpen : styles.boxSolid}
              data-status={box.status}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : null}
    </div>
  );
}
