"""Structured-output benchmark: extract an order from messy prose into schema-valid JSON.

Production LLM features (tool calling, form filling, ETL) fail in two distinct ways, and this task
measures them separately:

* **format** – is the output parseable, schema-valid JSON? (``json_valid`` / ``json_strict``)
* **content** – are the extracted values right? (``json_fields``, with partial credit)

Facts are shuffled, dates are written in varied human formats, prices sometimes appear in cents,
and the total must be *computed* – so the task can't be solved by copy-paste alone.
"""

from __future__ import annotations

import datetime as dt
import json
import random

from llmverdict.tasks.base import TaskSpec
from llmverdict.types import Message, Sample

SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "product": {"type": "string"},
        "quantity": {"type": "integer", "minimum": 1},
        "unit_price": {"type": "number"},
        "total": {"type": "number"},
        "express_shipping": {"type": "boolean"},
        "order_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
    "required": [
        "customer",
        "product",
        "quantity",
        "unit_price",
        "total",
        "express_shipping",
        "order_date",
    ],
    "additionalProperties": False,
}

CUSTOMERS = [
    "Anika Rao",
    "Tomás Herrera",
    "Grace Okafor",
    "Liam Chen",
    "Sofia Rossi",
    "Yusuf Demir",
]
PRODUCTS = [
    "wireless keyboard",
    "standing desk mat",
    "USB-C hub",
    "noise-cancelling headphones",
    "ergonomic chair",
    "27-inch monitor",
    "webcam cover pack",
]
OPENERS = ["Hi team,", "Hello,", "Good morning!", "Hey there -", "Dear support,"]


def _date_phrase(rng: random.Random, date: dt.date) -> str:
    day = date.day
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return rng.choice(
        [
            f"{date.strftime('%B')} {day}{suffix}, {date.year}",
            f"{day} {date.strftime('%b')} {date.year}",
            f"{date.month}/{day}/{date.year} (US format)",
            f"the {day}{suffix} of {date.strftime('%B')} {date.year}",
        ]
    )


def _price_phrase(rng: random.Random, price: float) -> str:
    cents = round(price * 100)
    return rng.choice([f"${price:.2f}", f"{cents} cents", f"USD {price:.2f}"])


def _render(rng: random.Random, fields: dict[str, object], date: dt.date) -> str:
    customer = str(fields["customer"])
    facts = [
        f"I'd like to order {fields['quantity']} units of the {fields['product']}.",
        f"Each one is listed at {_price_phrase(rng, float(str(fields['unit_price'])))}.",
        f"Please treat the order as placed on {_date_phrase(rng, date)}.",
        "I need it fast, so please use express shipping."
        if fields["express_shipping"]
        else "Standard shipping is fine, no rush.",
    ]
    rng.shuffle(facts)
    noise = rng.choice(
        [
            "By the way, last month's order of 2 desk lamps arrived fine.",
            "My colleague may also order 5 monitors next week, but that's separate.",
            "",
        ]
    )
    body = " ".join(f for f in [*facts, noise] if f)
    return f"{rng.choice(OPENERS)} {body} Thanks, {customer}"


def build(n: int = 60, seed: int = 0) -> TaskSpec:
    rng = random.Random(seed)
    samples = []
    schema_text = json.dumps(SCHEMA, indent=2)
    system = (
        "You convert customer messages into JSON for an order-processing system. "
        "Return ONLY a JSON object that conforms to this JSON Schema - no prose, no code fences.\n"
        f"{schema_text}\n"
        "Rules: unit_price and total are in US dollars; total = quantity * unit_price; "
        "order_date uses ISO format YYYY-MM-DD; ignore information about other orders."
    )
    for i in range(n):
        quantity = rng.randint(1, 12)
        unit_price = round(rng.uniform(4, 400), 2)
        date = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
        expected: dict[str, object] = {
            "customer": rng.choice(CUSTOMERS),
            "product": rng.choice(PRODUCTS),
            "quantity": quantity,
            "unit_price": unit_price,
            "total": round(quantity * unit_price, 2),
            "express_shipping": rng.random() < 0.5,
            "order_date": date.isoformat(),
        }
        samples.append(
            Sample(
                id=f"json-{seed}-{i:04d}",
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=_render(rng, expected, date)),
                ],
                reference={"schema": SCHEMA, "expected": expected},
                tags=["type:order_extraction"],
            )
        )
    return TaskSpec(
        name="json_extraction",
        description="Schema-constrained information extraction with computed fields",
        samples=samples,
        default_scorers=[
            {"type": "json_schema", "name": "json_valid"},
            {"type": "json_schema", "name": "json_strict", "options": {"strict": True}},
            {"type": "json_fields", "options": {"tol": 0.01}},
        ],
        primary_metric="json_fields",
    )
