"use client";

import { useEffect, useMemo, useState } from "react";
import { LaneMetric } from "@/components/charts/LaneMetric";
import { Court } from "@/components/court/Court";
import { EMPTY_SCENE, lastShotSpot } from "@/components/court/scene";
import { Evidence } from "@/components/evidence/Evidence";
import type { EvidenceGroups } from "@/components/evidence/groups";
import { copy, fill } from "@/lib/copy";
import { getBrowserSource } from "@/lib/data/browser";
import type { Possession, Shot } from "@/lib/types/contract";
import { ValuePanels, type StageLanes } from "./PossessionStage";
import styles from "./possession.module.css";

const text = copy.review.possession;

type Load = { status: "loading" } | { status: "error" } | { status: "ready"; possession: Possession };

/** The last shot of the possession, located or not: the one the shot card shows. */
function lastShot(possession: Possession): Shot | null {
  return possession.shots[possession.shots.length - 1] ?? null;
}

export interface PossessionStillProps {
  gameSlug: string;
  possession: number;
  /** The length of the possession in seconds, from the possession list: the EPV chart's time axis. */
  duration: number;
  lanes: StageLanes;
  evidence: EvidenceGroups;
}

/**
 * The possession on the public site (PLAN.md D60, D21). The SportVU source is marked not for redistribution, so
 * this copy of the site holds no player position: in place of the playback it draws an empty court of the half
 * the offense attacks, with the shot chart's mark when the shot has one, and says where the playback is. The
 * per-possession values stay: the EPV curve, the shot card with its xFG, and each lane's held-out metric.
 */
export function PossessionStill({ gameSlug, possession: possessionNumber, duration, lanes, evidence }: PossessionStillProps) {
  const [load, setLoad] = useState<Load>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    getBrowserSource()
      .possession(gameSlug, possessionNumber)
      .then(
        (possession) => {
          if (!cancelled) setLoad({ status: "ready", possession });
        },
        () => {
          if (!cancelled) setLoad({ status: "error" });
        },
      );
    return () => {
      cancelled = true;
    };
  }, [gameSlug, possessionNumber]);

  const possession = load.status === "ready" ? load.possession : null;
  // Only a shot chart location is left to mark: the public data holds no location from the tracking.
  const spot = useMemo(() => (possession ? lastShotSpot(possession) : null), [possession]);
  const scene = useMemo(() => ({ ...EMPTY_SCENE, markers: spot ? [{ kind: "spot" as const, ...spot }] : [] }), [spot]);

  if (load.status === "error") {
    return (
      <p role="alert" className="label">
        {text.court.error}
      </p>
    );
  }

  const end = Math.max(0, duration, possession?.epv?.t[possession.epv.t.length - 1] ?? 0);
  const shotTimes = possession ? possession.shots.map((item) => item.t).filter((t): t is number => t != null) : [];
  return (
    <div className={styles.stage} aria-busy={possession === null} data-testid="possession-stage" data-still="true">
      <figure className={styles.figure}>
        {/* The half view needs the rim the offense attacks, so the full court stands in until the data arrives. */}
        <Court
          scene={scene}
          view={possession ? "half" : "full"}
          attackX={possession?.attack_x ?? null}
          label={possession ? fill(text.still.label, { offense: possession.offense }) : text.court.loadingLabel}
          layers={{ labels: false }}
        />
        <figcaption className={styles.caption}>
          <span data-testid="still-note">{text.still.note}</span>
          {spot ? <span>{text.still.shot}</span> : null}
        </figcaption>
      </figure>

      {possession ? (
        <>
          <ValuePanels
            possession={possession}
            shot={lastShot(possession)}
            end={end}
            time={end}
            shotTimes={shotTimes}
            still
            lanes={lanes}
            evidence={evidence}
          />
          <div className="stack stack-tight small" data-testid="ghost-notes">
            <p>{text.ghosts.private}</p>
            <LaneMetric lane={lanes.ghosting} model={text.ghosts.model} />
          </div>
          <Evidence what={text.court.title} sources={evidence.positions} />
          {evidence.other.length > 0 ? <Evidence what={text.otherSources} sources={evidence.other} /> : null}
        </>
      ) : (
        <p role="status" className="small">
          {text.court.loading}
        </p>
      )}
    </div>
  );
}
