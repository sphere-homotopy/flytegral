"""Experimental compositional Fly decoder: concepts × scenes × rhetorical moves.

The fly's *sampled* token trajectory independently chooses the math concept,
a physical scene and a rhetorical mechanism. No model or LLM writes the text.
This is a hand-designed grammar, not a trained understanding of humor.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from fly_window.text.meme_engine import lexical_similarity

WORDS = re.compile(r"[a-z][a-z-]*", re.IGNORECASE)


@dataclass(frozen=True)
class MathConcept:
    key: str
    triggers: tuple[str, ...]
    axes: tuple[str, ...]
    fact: str
    label: str


@dataclass(frozen=True)
class FlyScene:
    key: str
    axes: tuple[str, ...]
    observation: str
    problem: str
    objection: str


CONCEPTS = (
    MathConcept("boundary", ("boundary", "topology", "frontier"), ('barrier',), "the boundary need not be part of the set", "a boundary"),
    MathConcept("open", ("open", "interior", "topological"), ('barrier',), "the empty set is open", "openness"),
    MathConcept("compact", ("compact", "closed", "bounded", "cover"), ('enclosure',), "every open cover of a compact space has a finite subcover", "compactness"),
    MathConcept("convergence", ("limit", "converge", "cauchy", "sequence"), ('approach',), "convergence is about getting arbitrarily close", "convergence"),
    MathConcept("recurrence", ("recurrent", "orbit", "dynamics", "ergodic"), ('recurrence',), "a recurrent point keeps returning arbitrarily near where it began", "recurrence"),
    MathConcept("fixed_point", ("fixed", "brouwer", "point"), ('fixed',), "a fixed point is sent to itself", "a fixed point"),
    MathConcept("measure_zero", ("lebesgue", "measure", "null"), ('precision',), "a single point has lebesgue measure zero", "measure zero"),
    MathConcept("independence", ("independent", "probability", "conditional"), ('independence',), "independence is a statement about joint probabilities", "independence"),
    MathConcept("optimization", ("optimize", "gradient", "minimum", "maximum"), ('optimization',), "a local minimum does not have to be a global minimum", "local optimization"),
    MathConcept("symmetry", ("group", "symmetry", "rotation", "automorphism"), ('symmetry',), "symmetry can preserve a problem without solving it", "symmetry"),
    MathConcept("equivalence", ("isomorphic", "equivalence", "homeomorphic"), ('identity',), "homeomorphic spaces can look completely different", "equivalence"),
    MathConcept("continuity", ("continuous", "smooth", "differentiable"), ('continuity',), "continuous motion does not require a smooth path", "continuity"),
    MathConcept("counting", ("count", "combinatorics", "cardinal", "finite"), ('counting',), "n+1 objects in n boxes force at least two objects into one box", "the pigeonhole principle"),
    MathConcept("expectation", ("expected", "average", "mean", "martingale"), ('expectation',), "an expectation is not a promise about one outcome", "expected value"),
    MathConcept("locality", ("local", "neighborhood", "sheaf"), ('locality',), "a neighborhood of a point need not be very small", "locality"),
    MathConcept("entropy", ("entropy", "information", "random"), ('stochastic',), "a random process can have highly predictable-looking stretches", "randomness"),
)

SCENES = (
    FlyScene("glass", ('barrier', 'approach'), "i have tried crossing the same window since morning", "a window", "the window remains unconvinced"),
    FlyScene("other_window", ('barrier', 'symmetry'), "the laboratory has two windows and neither opens", "two windows", "i checked both sides"),
    FlyScene("edge", ('barrier', 'approach'), "the light is visible from here but not reachable from here", "a lit window", "there is a physical obstruction that nobody mentioned"),
    FlyScene("jar", ('enclosure', 'identity'), "the scientist put me in a jar and wrote a label on it", "a jar", "he has not explained the lid"),
    FlyScene("lid", ('enclosure', 'identity'), "someone removed the lid and replaced it with a thinner lid", "a jar", "i regard this as a change of notation"),
    FlyScene("corners", ('recurrence', 'fixed', 'locality'), "i keep returning to the same corner of the glass", "a corner", "i have learned precisely nothing new"),
    FlyScene("lamp", ('recurrence', 'stochastic'), "i have visited the lamp fourteen times in one afternoon", "a lamp", "there are other parts of the room"),
    FlyScene("wall", ('barrier', 'fixed'), "every route i take ends at this wall", "a wall", "i am running out of routes"),
    FlyScene("swatter", ('precision', 'expectation'), "the swatter found the exact place i was sitting", "one point on a table", "it did not appear to need any approximation"),
    FlyScene("hands", ('independence',), "the scientist raises the swatter at irregular intervals", "a hand holding a swatter", "i have begun recording the intervals"),
    FlyScene("forecast", ('expectation',), "the humans say i will probably survive this experiment", "the next experiment", "they refuse to define probably"),
    FlyScene("banana", ('optimization', 'approach'), "the banana keeps being moved farther from the glass", "a banana", "the objective function keeps moving"),
    FlyScene("two_bananas", ('optimization',), "i had to choose between two bananas and one disappeared", "two bananas", "the experimenter is not a neutral observer"),
    FlyScene("crumbs", ('optimization',), "i found three crumbs and the humans called it a feast", "three crumbs", "i have questions about the units"),
    FlyScene("orbit", ('recurrence', 'continuity'), "my flight path went around the bulb and back to the window", "a lightbulb", "this was not a planned return"),
    FlyScene("headwind", ('continuity', 'stochastic'), "the air from the fan changes my direction every few seconds", "a fan", "the fan was not in the assumptions"),
    FlyScene("calendar", ('independence', 'recurrence', 'stochastic'), "every day the researcher repeats yesterday's experiment", "an experiment", "he keeps calling the results independent"),
    FlyScene("counter", ('expectation',), "the scientist counts every approach i make to the sugar", "a sugar cube", "i had not consented to being a statistic"),
    FlyScene("map", ('locality',), "i mapped every point on the inside of the jar", "a map of a jar", "there is still no mark for an exit"),
    FlyScene("corridor", ('barrier', 'approach', 'locality'), "i can see the next room through a perfectly transparent divider", "the next room", "transparency is a surprisingly weak property"),
    FlyScene("oscillation", ('recurrence', 'continuity', 'stochastic'), "i alternate between the lamp and the glass", "two locations", "the third option has yet to appear"),
    FlyScene("shadow", ('barrier',), "the flyswatter's shadow covers my entire route", "a shadow", "i am having difficulty measuring anything else"),
    FlyScene("paired_jars", ("counting",), "the scientist put three flies in two jars", "three flies and two jars", "i volunteered to be the one without a roommate"),
    FlyScene("mirror", ("symmetry",), "the left and right walls reflect the lamp in the same way", "two reflections", "i collided with both"),
    FlyScene("trap_repeats", ("independence",), "the scientist reran the trap experiment with exactly the same trap", "a repeating trap", "the traps look suspiciously correlated"),
    FlyScene("needle", ("precision",), "a needle came down on the one crumb i was holding", "the crumb", "i would like the precision to be applied elsewhere"),
    FlyScene("still_spot", ("fixed",), "the scientist puts a pencil mark wherever my flight ends", "a pencil mark", "the mark has not moved once"),
    FlyScene("labelled_jars", ("identity",), "two identical jars have different names on their labels", "two labeled jars", "the labels matter more to the humans than the lids"),
    FlyScene("smaller", ("locality",), "i asked for a smaller neighborhood and the scientist moved my jar closer to his desk", "a neighborhood", "this does not appear to change the topology"),
    FlyScene("unpredictable", ("stochastic",), "the fan changes direction at random but stops every time i approach the ceiling", "a fan", "i am beginning to predict the randomness"),
    FlyScene("near_banana", ("approach",), "each flight gets me halfway closer to the banana behind the glass", "half the distance", "the remaining distance is still made of glass"),
    FlyScene("empty_cover", ("enclosure",), "the researcher has covered the jar with a sheet of paper", "a cover", "it does not have a visible subcover"),
)

MOVES = ("report", "objection", "label", "question", "misclassification", "diagnosis")

def _choice(seq: list[int], label: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("empty choice")
    # The seed is ONLY the sampled token segment from the actual fly.
    data = json.dumps([label, seq], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(data, digest_size=8).digest(), "big") % modulo


def _sampled_tokens(row: dict) -> list[int]:
    raw = row.get("token_ids")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw or not all(type(x) is int and 0 <= x < 65536 for x in raw):
        raise ValueError("requires nonempty actual Fly token_ids")
    return raw


def choose_parts(row: dict, *, topic_prior=None) -> tuple[MathConcept, FlyScene, str, str]:
    tokens = _sampled_tokens(row)
    # Disjoint streams: changing an environment token does not change math choice.
    math_stream, scene_stream, move_stream = tokens[0::3], tokens[1::3], tokens[2::3]
    math_candidates = list(CONCEPTS)
    words = WORDS.findall(str(row.get("text", "")).lower())
    salient = next((word for word in words if any(word in c.triggers for c in CONCEPTS)), None)
    if salient:
        math_candidates = [c for c in CONCEPTS if salient in c.triggers]
    if topic_prior is None or salient is not None:
        concept = math_candidates[_choice(math_stream, "math", len(math_candidates))]
    else:
        from fly_window.text.topic_prior import weighted_index
        sample = _choice(math_stream, "topic_prior", 1000000) / 1000000.
        concept = math_candidates[weighted_index([c.key for c in math_candidates], topic_prior, sample)]
    allowed_scenes = [scene for scene in SCENES if set(concept.axes).intersection(scene.axes)]
    scene = allowed_scenes[_choice(scene_stream, "scene", len(allowed_scenes))]
    move = MOVES[_choice(move_stream, "move", len(MOVES))]
    return concept, scene, move, salient or "token_stream"


def render_parts(concept: MathConcept, scene: FlyScene, move: str) -> str:
    observation, objection, fact = scene.observation, scene.objection, concept.fact
    if move == "report":
        return f"field notes: {observation}. apparently this is {concept.label}. {objection}."
    if move == "objection":
        return f"{fact}. {observation}. {objection}."
    if move == "label":
        return f"{observation}. the scientist wrote '{concept.label}' in his notebook. {objection}."
    if move == "question":
        return f"{observation}. {fact}. does this count as {concept.label}?"
    if move == "misclassification":
        return f"{observation}. i was told this is {concept.label}. {objection}."
    if move == "diagnosis":
        return f"i have been studying {concept.label}. {observation}. {objection}."
    raise ValueError("unrecognized move")


def compose_fly_trace(row: dict, *, max_chars: int = 280) -> dict:
    if max_chars < 60:
        raise ValueError("max_chars must be >= 60")
    concept, scene, initial_move, trigger = choose_parts(row)
    start = MOVES.index(initial_move)
    for i in range(len(MOVES)):
        move = MOVES[(start + i) % len(MOVES)]
        text = render_parts(concept, scene, move)
        if len(text) <= max_chars:
            return {
                "text": text,
                "decoder": "fly_compositional_v1",
                "fly_topic": concept.key,
                "fly_scene": scene.key,
                "fly_move": move,
                "fly_trigger": trigger,
                "source_idempotency_key": row.get("idempotency_key", ""),
                "publishable": False,
            }
    raise ValueError("no complete rendering fits limit")


def select_compositions(rows: list[dict], count: int = 32, *, max_chars: int = 280) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    collected: list[dict] = []
    used: set[str] = set()
    for row in rows:
        try:
            candidate = compose_fly_trace(row, max_chars=max_chars)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        key = candidate["source_idempotency_key"]
        if key in used:
            continue
        if any(lexical_similarity(candidate["text"], other["text"]) >= 0.73 for other in collected):
            continue
        used.add(key)
        collected.append(candidate)
        if len(collected) >= count:
            break
    return collected
