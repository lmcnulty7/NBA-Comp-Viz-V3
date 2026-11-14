import { Status, type StatusTone } from "@/components/ui/Status";
import { copy, fill, label } from "@/lib/copy";
import type { Scorecard, ScorecardBuild, ScorecardCell } from "@/lib/types/contract";
import { DataText, DataValue } from "./DataValue";
import { Sources } from "./Sources";

const text = copy.results;
const FILE = "scorecard.json";

const STATUS_TONE: Record<ScorecardCell["status"], StatusTone> = {
  met: "pass",
  "not met": "fail",
  reported: "plain",
  "not measurable": "plain",
};

/** A data key such as position_error as words. */
export function words(key: string): string {
  return key.replace(/_/g, " ");
}

function Cell({ build, rowKey, cell }: { build: string; rowKey: string; cell: ScorecardCell }) {
  const path = `builds[build=${build}].rows[key=${rowKey}].cells[game=${cell.game}]`;
  return (
    <td className="wrap" data-status={cell.status}>
      <Status tone={STATUS_TONE[cell.status]} text={label(text.scorecard.status, cell.status)} />
      <DataText className="note" file={FILE} field={`${path}.text`} text={cell.text} />
      <span className="note">
        {cell.interval ? (
          <DataText file={FILE} field={`${path}.interval`} text={cell.interval} />
        ) : (
          copy.site.estimate.point
        )}
        {cell.n != null ? (
          <>
            {", "}
            <DataValue file={FILE} field={`${path}.n`} value={cell.n} shown="count, as stored" />{" "}
            {cell.n_unit ?? ""}
          </>
        ) : null}
      </span>
    </td>
  );
}

function BuildSection({ build }: { build: ScorecardBuild }) {
  const games = build.games.map((game) => game.game);
  const caption = fill(text.scorecard.caption, { build: build.build });
  const gamesCaption = fill(text.scorecard.games.caption, { build: build.build });
  return (
    <div className="stack" data-build={build.build} data-role={build.role}>
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
              {build.games.map((game) => (
                <th scope="col" key={game.game}>
                  {words(game.game)}
                  <span className="note">{label(text.scorecard.split, game.split)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {build.rows.map((row) => (
              <tr key={row.key} data-row={row.key}>
                <th scope="row">{words(row.key)}</th>
                <td className="wrap">
                  <DataText file={FILE} field={`builds[build=${build.build}].rows[key=${row.key}].target`} text={row.target} />
                </td>
                {games.map((game) => {
                  const cell = row.cells.find((candidate) => candidate.game === game);
                  return cell ? (
                    <Cell key={game} build={build.build} rowKey={row.key} cell={cell} />
                  ) : (
                    <td key={game} className="muted">
                      {text.scorecard.notScored}
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
            {build.games.map((game) => {
              const path = `builds[build=${build.build}].games[game=${game.game}]`;
              return (
                <tr key={game.game}>
                  <th scope="row">{words(game.game)}</th>
                  <td>{label(text.scorecard.split, game.split)}</td>
                  <td>
                    <DataValue file={FILE} field={`${path}.truth_frames`} value={game.truth_frames} shown="count, as stored" />
                  </td>
                  <td>
                    <DataValue file={FILE} field={`${path}.testable`} value={game.testable} shown="count, as stored" />
                  </td>
                  <td>
                    <DataValue
                      file={FILE}
                      field={`${path}.untestable_share`}
                      value={game.untestable_share}
                      shown="share as a percentage"
                    />
                    {text.scorecard.percentSign}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {build.caveats.length > 0 ? (
        <div className="stack stack-tight column">
          <h4 className="eyebrow">{text.scorecard.caveats}</h4>
          <ul className="bullets small">
            {build.caveats.map((caveat, index) => (
              <li key={caveat}>
                <DataText file={FILE} field={`builds[build=${build.build}].caveats[${index}]`} text={caveat} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/**
 * The tracking scorecard against the Stage 1 targets (C5): the adopted build first, then each measured
 * candidate with the rows the data gives it, then the paired differences between builds. Every status is the
 * one the data states.
 */
export function ScorecardSection({ scorecard }: { scorecard: Scorecard }) {
  const builds = [...scorecard.builds].sort((a, b) => Number(b.role === "adopted") - Number(a.role === "adopted"));
  const method = Object.entries(scorecard.method);
  return (
    <section className="section" aria-labelledby="scorecard" data-testid="scorecard">
      <div className="column stack stack-tight">
        <h2 id="scorecard">{text.scorecard.title}</h2>
        <p className="small">{text.scorecard.intro}</p>
        <p>
          <strong>{text.scorecard.decision}.</strong>{" "}
          <DataText file={FILE} field="decision" text={scorecard.decision} />
        </p>
      </div>
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
                {scorecard.paired.map((pair, index) => {
                  const path = `paired[${index}]`;
                  return (
                    <tr key={`${pair.game}|${pair.metric}|${pair.baseline}|${pair.build}`} data-paired={index}>
                      <th scope="row">{words(pair.game)}</th>
                      <td>{words(pair.metric)}</td>
                      <td>{pair.baseline}</td>
                      <td>{pair.build}</td>
                      <td>
                        <DataValue file={FILE} field={`${path}.diff`} value={pair.diff} signed />
                      </td>
                      <td>
                        <DataValue file={FILE} field={`${path}.lo`} value={pair.lo} signed />
                        {text.scorecard.paired.to}
                        <DataValue file={FILE} field={`${path}.hi`} value={pair.hi} signed />
                      </td>
                      <td>{pair.excludes_zero ? copy.site.yes : copy.site.no}</td>
                    </tr>
                  );
                })}
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
                  <DataText file={FILE} field={`method[${key}]`} text={value} />
                </li>
              ))}
            </ul>
          </>
        ) : null}
        <Sources sources={scorecard.sources} />
      </div>
    </section>
  );
}
