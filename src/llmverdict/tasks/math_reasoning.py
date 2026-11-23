"""Procedurally generated multi-step reasoning problems.

Why procedural? Static public benchmarks leak into pre-training corpora, so a model can score
well by memorisation. Fresh instances drawn from a seeded generator are contamination-resistant,
have exact ground truth computed in code, and can be scaled in difficulty on demand. Fixing the
seed keeps runs reproducible and comparable.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable

from llmverdict.tasks.base import TaskSpec
from llmverdict.types import Message, Sample

INSTRUCTIONS = (
    "Solve the problem. Reason step by step, then end with a final line of the form "
    "'Answer: <value>'. Give numbers without units or currency symbols."
)

NAMES = ["Aisha", "Ravi", "Mei", "Lucas", "Priya", "Omar", "Elena", "Kenji", "Zara", "Diego"]
ITEMS = [("notebooks", 2.5, 9.0), ("pens", 0.75, 3.5), ("mugs", 4.0, 15.0), ("lamps", 12.0, 45.0)]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

Problem = tuple[str, float | int | str]
Generator = Callable[[random.Random, int], Problem]


def _price(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi) * 4) / 4  # quarter-dollar prices keep arithmetic exact


def shopping(rng: random.Random, level: int) -> Problem:
    name = rng.choice(NAMES)
    (i1, lo1, hi1), (i2, lo2, hi2) = rng.sample(ITEMS, 2)
    q1, q2 = rng.randint(2, 4 * level + 3), rng.randint(2, 4 * level + 3)
    p1, p2 = _price(rng, lo1, hi1), _price(rng, lo2, hi2)
    discount = rng.choice([5, 10, 15, 20, 25])
    total = (q1 * p1 + q2 * p2) * (1 - discount / 100)
    text = (
        f"{name} buys {q1} {i1} at ${p1:.2f} each and {q2} {i2} at ${p2:.2f} each. "
        f"A {discount}% discount is applied to the whole order. How much does {name} pay?"
    )
    if level >= 2:
        tax = rng.choice([5, 8, 12])
        total *= 1 + tax / 100
        text = text.replace("How much", f"Then {tax}% sales tax is added. How much")
    return text, round(total, 2)


def inventory(rng: random.Random, level: int) -> Problem:
    name = rng.choice(NAMES)
    start = rng.randint(20, 60 * level)
    given = rng.randint(3, start // 2)
    packs, per_pack = rng.randint(2, 3 + level), rng.randint(4, 12)
    answer = start - given + packs * per_pack
    distractor = f" Their cousin owns {rng.randint(5, 90)} marbles." if level >= 2 else ""
    text = (
        f"{name} has {start} marbles and gives {given} of them to a friend.{distractor} "
        f"Then {name} buys {packs} packs with {per_pack} marbles in each pack. "
        f"How many marbles does {name} have now?"
    )
    if level >= 3:
        lost = rng.randint(1, 9)
        answer -= lost
        text = text.replace("How many", f"On the way home {lost} marbles are lost. How many")
    return text, answer


def work_rate(rng: random.Random, level: int) -> Problem:
    rate = rng.randint(3, 15 * level)
    hours, minutes = rng.randint(1, 2 + level), rng.choice([0, 15, 20, 30, 45])
    downtime = rng.choice([0, 10, 25]) if level >= 2 else 0
    working = hours * 60 + minutes - downtime
    text = (
        f"A machine fills {rate} bottles per minute. "
        f"It runs for {hours} hours and {minutes} minutes"
    )
    text += (
        f", but is stopped for maintenance for {downtime} minutes of that time."
        if downtime
        else "."
    )
    return text + " How many bottles does it fill?", rate * working


def percent_chain(rng: random.Random, level: int) -> Problem:
    price = rng.randint(40, 400)
    changes = [rng.choice([-30, -20, -10, 10, 20, 25, 50]) for _ in range(1 + level)]
    value = float(price)
    parts = []
    for change in changes:
        value *= 1 + change / 100
        parts.append(f"{'rises' if change > 0 else 'falls'} by {abs(change)}%")
    text = (
        f"A jacket costs ${price}. Its price then {', then '.join(parts)}. "
        "What is the final price? Round to 2 decimal places."
    )
    return text, round(value, 2)


def leftover(rng: random.Random, level: int) -> Problem:
    groups = rng.randint(3, 8 + level)
    students = groups * rng.randint(3, 12 * level) + rng.randint(1, groups - 1)
    text = (
        f"A teacher splits {students} students into teams of exactly {groups}. "
        "How many students are left over without a full team?"
    )
    return text, students % groups


def weekday_offset(rng: random.Random, level: int) -> Problem:
    start = dt.date(2024, 1, 1) + dt.timedelta(days=rng.randint(0, 1500))
    offset = rng.randint(10, 100 * level)
    target = start + dt.timedelta(days=offset)
    text = (
        f"Today is {WEEKDAYS[start.weekday()]}, {start.strftime('%B')} {start.day}, {start.year}. "
        f"What day of the week will it be {offset} days from today?"
    )
    return text, WEEKDAYS[target.weekday()]


GENERATORS: dict[str, Generator] = {
    "shopping": shopping,
    "inventory": inventory,
    "work_rate": work_rate,
    "percent_chain": percent_chain,
    "leftover": leftover,
    "weekday_offset": weekday_offset,
}


def build(n: int = 120, seed: int = 0, levels: tuple[int, ...] = (1, 2, 3)) -> TaskSpec:
    rng = random.Random(seed)
    kinds = sorted(GENERATORS)
    samples = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        level = levels[(i // len(kinds)) % len(levels)]
        question, answer = GENERATORS[kind](rng, level)
        samples.append(
            Sample(
                id=f"math-{seed}-{i:04d}",
                messages=[
                    Message(role="system", content=INSTRUCTIONS),
                    Message(role="user", content=question),
                ],
                reference=answer,
                tags=[f"type:{kind}", f"level:{level}"],
            )
        )
    return TaskSpec(
        name="math_reasoning",
        description="Procedural multi-step arithmetic & date reasoning (contamination-resistant)",
        samples=samples,
        # 1-cent tolerance: "round to 2 dp" is ambiguous at exact half-cents (banker's vs half-up).
        default_scorers=[{"type": "final_answer", "options": {"abs_tol": 0.0100001}}],
        primary_metric="final_answer",
    )
