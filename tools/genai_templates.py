"""tools/genai_templates.py: the template writer of genai-E1 (docs/research/genai.md), no model.

A text is a fixed list of sentences, each filled from the item's facts and context exactly as export.genai builds
them: a sentence is written only when every fact it needs is present, and each number is the fact's `shown` text.
Nothing is chosen at random and nothing depends on another item, so the same item always gets the same text.

The sentences were written and checked on train items only (tools.genai_data items "train"), before genai-E1 read a
test item. They follow the rules of export/prompts (one paragraph, digits only, a value with its range, "point
estimate" for a point estimate, a model word in the sentence of a model value, the status of a lane that did not work,
descriptive values called descriptive, short plain sentences). The checker (export.genai.check_text) is not called
here: the evaluation calls it on the output, as it does on every other arm.
"""

from export.genai import STATUS_WORDS

SHOT_KIND = {"three-point": "three-point", "two-point": "two-point", None: ""}
POSITION = {"G": "guard", "F": "forward", "C": "center"}


def _facts(item: dict) -> dict:
    return {fact["field"]: fact["shown"] for fact in item["facts"]}


def _status(item: dict, metric: str) -> str:
    words = STATUS_WORDS.get(item["context"].get("model_status", {}).get(metric, "works"))
    return f" Its model has the status {words}." if words else ""


def _range(f: dict, field: str, unit: str = "") -> str:
    """ "(95% range LO to HI)" for an estimate with an interval, without the level when the page has none."""
    level = f.get(f"{field}.interval")
    tail = f" {unit}" if unit else ""
    return f"({level + ' ' if level else ''}range {f[field + '.lo']} to {f[field + '.hi']}{tail})"


def _names(names: list) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def scouting_text(item: dict) -> str:
    f, c = _facts(item), item["context"]
    shown = c.get("shown", {})
    out = []
    role = POSITION.get(c.get("position"))
    teams = _names(c["teams"]) if c.get("teams") else ""
    who = f"{c['name']} played {f['games']} tracked games"
    who += f" for {teams}" if teams else ""
    out.append(who + (f" as a {role}." if role else "."))
    point = False

    if "shot_making.over_expected_per_100.value" in f:
        g = "shot_making.over_expected_per_100"
        out.append(f"He made {f[g + '.value']} more shots per 100 tries than the model expected "
                   f"{_range(f, g)}." + _status(item, "shot_making"))
        if "shot_making.fg_pct" in f and "shot_making.expected_fg_pct" in f:
            out.append(f"Of his shots, {f['shot_making.fg_pct']} went in, and the model expected "
                       f"{f['shot_making.expected_fg_pct']}.")
            point = True
    elif not shown.get("shot_making"):
        out.append(f"His shot value is not shown: he took {f['shot_making.attempts']} shots, and the page needs "
                   f"{f['shot_making.min_attempts']}.")

    if "gravity.gravity_ft.value" in f:
        g = "gravity.gravity_ft"
        out.append(f"Defenders stay {f[g + '.value']} feet closer to him than his spot would tell "
                   f"{_range(f, g, 'feet')}, which is descriptive. A minus sign means more room."
                   + _status(item, "gravity"))
    elif not shown.get("gravity"):
        out.append(f"How close defenders stay is not shown: he has {f['gravity.possessions']} trips, and the page "
                   f"needs {f['gravity.min_possessions']}.")

    if "rebounding.rate_per_100.value" in f:
        g = "rebounding.rate_per_100"
        out.append(f"He got {f[g + '.value']} rebounds per 100 chances {_range(f, g)}, which is descriptive."
                   + _status(item, "rebounding"))
    elif not shown.get("rebounding"):
        out.append(f"His rebound rate is not shown: he had {f['rebounding.chances']} chances, and the page needs "
                   f"{f['rebounding.min_chances']}.")

    if "absence_risk.model_probability.value" in f:
        g = "absence_risk.model_probability"
        out.append(f"The model gives a {f[g + '.value']} chance that he will miss the next tracked game "
                   f"{_range(f, g)}. This is not an injury forecast." + _status(item, "absence_risk"))
    elif not shown.get("absence_risk"):
        out.append(f"His chance to miss a game is not shown: he has {f['absence_risk.games']} games, and the page "
                   f"needs {f['absence_risk.min_games']}.")

    similar = c.get("similar_players") or []
    if similar:
        out.append(f"{similar[0]} is the player most like him in style.")
    if point:
        out.append("Each number with no range is a point estimate.")
    return " ".join(out)


def possession_text(item: dict) -> str:
    f, c = _facts(item), item["context"]
    out = [f"{c['offense']} had the ball against {c['defense']} in period {f['period']}, with {f['start_clock']} on "
           "the game clock."]
    points = f["points"]
    unit = "point" if points == "1" else "points"
    if c["result"] == "other end":
        out.append(f"The play lasted {f['duration_s']} seconds and was worth {points} {unit}.")
    elif c["result"] == "end of the period":
        out.append(f"The play lasted {f['duration_s']} seconds and ran to the end of the period, worth {points} "
                   f"{unit}.")
    else:
        out.append(f"The play lasted {f['duration_s']} seconds and ended in {c['result']}, worth {points} {unit}."
                   if c["result"] == "free throws" else
                   f"The play lasted {f['duration_s']} seconds and ended in a {c['result']}, worth {points} {unit}.")
    point = False
    shots = c.get("shots") or []
    for j, shot in enumerate(shots):
        kind = SHOT_KIND.get(shot.get("kind"), "")
        who = shot.get("shooter") or "A player"
        what = f"a {kind} shot" if kind else "a shot"
        verb = "made" if shot.get("made") else "missed"
        sentence = f"{who} took {what} and {verb} it."
        if f"shots.{j}.xfg" in f:
            sentence += f" The model gave it a {f[f'shots.{j}.xfg']} chance to go in."
            point = True
        out.append(sentence)
    if "epv_start" in f and "epv_end" in f:
        out.append(f"The model expected {f['epv_start']} points at the start and {f['epv_end']} points at the end.")
        point = True
    if c.get("clock_gap"):
        out.append("The game clock has a gap in this play.")
    if point:
        out.append("Each model number is a point estimate.")
        out[-1] += _status(item, "xfg") if any(k.startswith("shots.") or k == "xfg" for k in f) else ""
        out[-1] += _status(item, "epv") if "epv_start" in f else ""
    if len(" ".join(out).split()) < 34:
        out.append(f"This was trip {f['possession']} of the game for the two teams.")
    return " ".join(out)


def template_text(item: dict) -> str:
    return scouting_text(item) if item["kind"] == "scouting" else possession_text(item)
