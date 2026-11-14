import type { Metadata } from "next";
import { SourceList } from "@/components/ui/SourceList";
import { Status, type StatusTone } from "@/components/ui/Status";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount, formatPercent, formatSigned } from "@/lib/format";
import type { LaneResult, ScorecardBuild, ScorecardCell } from "@/lib/types/contract";

const text = copy.results;

export const metadata: Metadata = { title: text.title };

const STATUS_TONE: Record<ScorecardCell["status"], StatusTone> = {
  met: "pass",
  "not met": "fail",
  reported: "plain",
  "not measurable": "plain",
};

const VERDICT_TONE: Record<LaneResult["verdict_class"], StatusTone> = {
  works: "pass",
  no_signal: "plain",
  failed: "fail",
  inconclusive: "warn",
};

/** A data key such as position_error as words. */
function words(key: string): string {
  return key.replace(/_/g, " ");
}

function BuildSection({ build }: { build: ScorecardBuild }) {
  const games = build.games.map((game) => game.game);
  const caption = fill(text.scorecard.caption, { build: build.build });
  const gamesCaption = fill(text.scorecard.games.caption, { build: build.build });
  return (
    <div className="stack" data-build={build.build}>
      <div className="stack stack-tight column">
        <h3>{fill(text.scorecard.build, { build: build.build })}</h3>
        <p className="label">
          <strong>{label(text.scorecard.role, build.role)}.</strong> {build.note}
        </p>
      </div>
      <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>
              <th scope="col">{text.scorecard.row}</th>
              <th scope="col">{text.scorecard.target}</th>
              {games.map((game) => (
                <th scope="col" key={game}>
                  {words(game)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {build.rows.map((row) => (
              <tr key={row.key}>
                <th scope="row">{words(row.key)}</th>
                <td className="wrap">{row.target}</td>
                {games.map((game) => {
                  const cell = row.cells.find((candidate) => candidate.game === game);
                  if (!cell) return <td key={game} />;
                  return (
                    <td key={game} className="wrap">
                      <Status tone={STATUS_TONE[cell.status]} text={label(text.scorecard.status, cell.status)} />
                      <span className="note num">{cell.text}</span>
                      <span className="note">
                        {cell.interval ?? copy.site.estimate.point}
                        {cell.n != null ? (
                          <>
                            {", "}
                            {fill(text.scorecard.sample, { n: formatCount(cell.n), unit: cell.n_unit ?? "" }).trim()}
                          </>
                        ) : null}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-wrap" role="region" aria-label={gamesCaption} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{gamesCaption}</caption>
          <thead>
            <tr>
              <th scope="col">{text.scorecard.games.game}</th>
              <th scope="col">{text.scorecard.games.split}</th>
              <th scope="col">{text.scorecard.games.truthFrames}</th>
              <th scope="col">{text.scorecard.games.testable}</th>
              <th scope="col">{text.scorecard.games.untestable}</th>
            </tr>
          </thead>
          <tbody>
            {build.games.map((game) => (
              <tr key={game.game}>
                <th scope="row">{words(game.game)}</th>
                <td>{label(copy.review.split, game.split)}</td>
                <td className="num">{formatCount(game.truth_frames)}</td>
                <td className="num">{formatCount(game.testable)}</td>
                <td className="num">{formatPercent(game.untestable_share, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {build.caveats.length > 0 ? (
        <div className="stack stack-tight column">
          <h4 className="eyebrow">{text.scorecard.caveats}</h4>
          <ul className="bullets small">
            {build.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export default async function ResultsPage() {
  const source = getServerSource();
  const [scorecard, results] = await Promise.all([source.scorecard(), source.results()]);
  // The adopted build first, then the measured candidates.
  const builds = [...scorecard.builds].sort((a, b) => Number(b.role === "adopted") - Number(a.role === "adopted"));
  const method = Object.entries(scorecard.method);
  return (
    <div className="wide stack stack-loose">
      <header className="page-head column">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
      </header>

      <section className="section" aria-labelledby="scorecard">
        <h2 id="scorecard" className="column">
          {text.scorecard.title}
        </h2>
        <p className="column">
          <strong>{text.scorecard.decision}.</strong> {scorecard.decision}
        </p>
        {builds.map((build) => (
          <BuildSection key={build.build} build={build} />
        ))}

        {scorecard.paired.length > 0 ? (
          <div className="stack">
            <div className="stack stack-tight column">
              <h3>{text.scorecard.paired.title}</h3>
              <p className="small">{text.scorecard.paired.note}</p>
            </div>
            <div className="table-wrap" role="region" aria-label={text.scorecard.paired.caption} tabIndex={0}>
              <table>
                <caption className="visually-hidden">{text.scorecard.paired.caption}</caption>
                <thead>
                  <tr>
                    <th scope="col">{text.scorecard.paired.game}</th>
                    <th scope="col">{text.scorecard.paired.metric}</th>
                    <th scope="col">{text.scorecard.paired.baseline}</th>
                    <th scope="col">{text.scorecard.paired.build}</th>
                    <th scope="col">{text.scorecard.paired.diff}</th>
                    <th scope="col">{text.scorecard.paired.interval}</th>
                    <th scope="col">{text.scorecard.paired.excludesZero}</th>
                  </tr>
                </thead>
                <tbody>
                  {scorecard.paired.map((pair) => (
                    <tr key={`${pair.game}|${pair.metric}|${pair.baseline}|${pair.build}`}>
                      <th scope="row">{words(pair.game)}</th>
                      <td>{words(pair.metric)}</td>
                      <td>{pair.baseline}</td>
                      <td>{pair.build}</td>
                      <td className="num">{formatSigned(pair.diff, 2)}</td>
                      <td className="num">
                        {fill(text.scorecard.paired.range, {
                          lo: formatSigned(pair.lo, 2),
                          hi: formatSigned(pair.hi, 2),
                        })}
                      </td>
                      <td>{pair.excludes_zero ? copy.site.yes : copy.site.no}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        <div className="column stack stack-tight">
          {method.length > 0 ? (
            <>
              <h3>{text.scorecard.method}</h3>
              <ul className="bullets small">
                {method.map(([key, value]) => (
                  <li key={key}>
                    {words(key)}
                    {": "}
                    <span className="num">{value}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
          <SourceList sources={scorecard.sources} />
        </div>
      </section>

      <section className="section" aria-labelledby="models">
        <div className="column stack stack-tight">
          <h2 id="models">{text.models.title}</h2>
          <p className="small">{fill(text.models.frozen, { tag: results.freeze_tag, when: results.generated })}</p>
        </div>
        <div className="table-wrap" role="region" aria-label={text.models.caption} tabIndex={0}>
          <table>
            <caption className="visually-hidden">{text.models.caption}</caption>
            <thead>
              <tr>
                <th scope="col">{text.models.lane}</th>
                <th scope="col">{text.models.question}</th>
                <th scope="col">{text.models.selected}</th>
                <th scope="col">{text.models.oof}</th>
                <th scope="col">{text.models.heldout}</th>
                <th scope="col">{text.models.verdict}</th>
              </tr>
            </thead>
            <tbody>
              {results.lanes.map((lane) => (
                <tr key={lane.lane} data-verdict={lane.verdict_class}>
                  <th scope="row">
                    {words(lane.lane)}
                    <span className="note">
                      {text.models.report} <code>{lane.report}</code>
                    </span>
                  </th>
                  <td className="wrap">{lane.question}</td>
                  <td className="wrap">{lane.selected_model}</td>
                  <td className="wrap">
                    <span className="num">{lane.out_of_fold.text}</span>
                    {lane.out_of_fold.point_estimate ? (
                      <span className="note">{copy.site.estimate.point}</span>
                    ) : null}
                  </td>
                  <td className="wrap">
                    {lane.heldout ? (
                      <>
                        <span className="num">{lane.heldout.text}</span>
                        {lane.heldout.point_estimate ? (
                          <span className="note">{copy.site.estimate.point}</span>
                        ) : null}
                      </>
                    ) : (
                      <span className="muted">{text.models.noHeldout}</span>
                    )}
                  </td>
                  <td className="wrap">
                    <Status
                      tone={VERDICT_TONE[lane.verdict_class]}
                      text={label(text.models.verdictClass, lane.verdict_class)}
                    />
                    <q className="note">{lane.verdict}</q>
                    <span className="note">{fill(text.models.quoted, { source: lane.verdict_source })}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="column stack">
          {results.pending.length > 0 ? (
            <div className="stack stack-tight">
              <h3>{text.models.pending.title}</h3>
              <ul className="bullets small">
                {results.pending.map((pending) => (
                  <li key={pending.lane}>
                    {fill(text.models.pending.row, { lane: words(pending.lane), reason: pending.reason })}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {results.notes.length > 0 ? (
            <div className="stack stack-tight">
              <h3>{text.models.notes}</h3>
              <ul className="bullets small">
                {results.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
          ) : null}
          <SourceList sources={results.sources} />
        </div>
      </section>
    </div>
  );
}
