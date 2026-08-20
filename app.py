import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# Regular expressions
# ============================================================

# GCS-style URI: gs://bucket/object
URI_RE = re.compile(r"^gs://[^/]+/.+$")

# "Decimal string" means one or more decimal digits.
# Leading zeroes are allowed.
GENERATION_RE = re.compile(r"^[0-9]+$")

# Exactly 8 lowercase hexadecimal characters.
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)


# ============================================================
# Deterministic JSON helpers
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def sorted_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=utf8)


def add_code(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


# ============================================================
# Timestamp handling
# ============================================================

def parse_timestamp(value: Any):
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year_s,
        month_s,
        day_s,
        hour_s,
        minute_s,
        second_s,
        fraction,
        offset,
    ) = match.groups()

    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    hour = int(hour_s)
    minute = int(minute_s)
    second = int(second_s)

    # Validate timezone offset.
    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # ±14:00 is valid, ±14:01 is not.
        if offset_hour == 14 and offset_minute != 0:
            return None

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            tzinfo=tz,
        )
    except ValueError:
        return None

    if fraction is None:
        milliseconds = 0
    elif len(fraction) == 1:
        milliseconds = int(fraction) * 100
    elif len(fraction) == 2:
        milliseconds = int(fraction) * 10
    else:
        milliseconds = int(fraction)

    dt = dt.replace(
        microsecond=milliseconds * 1000
    )

    dt = dt.astimezone(timezone.utc)

    canonical = (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{dt.microsecond // 1000:03d}Z"
    )

    return dt, canonical


# ============================================================
# Unicode canonicalization
# ============================================================

def canonicalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    return " ".join(value.split())


# ============================================================
# CRC32C Castagnoli
# ============================================================

def crc32c(data: bytes) -> int:
    polynomial = 0x82F63B78

    table = []

    for i in range(256):
        crc = i

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1

        table.append(crc)

    crc = 0xFFFFFFFF

    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


# ============================================================
# Primitive validation
# ============================================================

def valid_generation(value: Any) -> bool:
    return (
        isinstance(value, str)
        and GENERATION_RE.fullmatch(value) is not None
    )


def valid_crc_syntax(value: Any) -> bool:
    return (
        isinstance(value, str)
        and CRC32C_RE.fullmatch(value) is not None
    )


def valid_revision(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


# ============================================================
# Contamination
# ============================================================

def word_set(value: str) -> set[str]:
    words = set()
    current = []

    for ch in value.lower():
        category = unicodedata.category(ch)

        if category.startswith("L") or category.startswith("N"):
            current.append(ch)
        else:
            if current:
                words.add("".join(current))
                current = []

    if current:
        words.add("".join(current))

    return words


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# Split
# ============================================================

def bucket(entity: str) -> int:
    digest = hashlib.sha256(
        entity.encode("utf-8")
    ).digest()

    return digest[0] % 10


def split_name(value: int) -> str:
    if value <= 5:
        return "train"

    if value <= 7:
        return "validation"

    return "test"


# ============================================================
# Response sorting
# ============================================================

def sort_rejected_objects(items):
    return sorted(
        items,
        key=lambda item: (
            utf8(item["uri"])
            if isinstance(item["uri"], str)
            else b"",
            utf8(compact_json(item)),
        ),
    )


def sort_rejected_rows(items):
    return sorted(
        items,
        key=lambda item: (
            utf8(item["id"]),
            utf8(compact_json(item)),
        ),
    )


def sort_lineage(items):
    return sorted(
        items,
        key=lambda item: (
            utf8(item["uri"]),
            utf8(compact_json(item)),
        ),
    )


# ============================================================
# Invalid request
# ============================================================

def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


# ============================================================
# Endpoint
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    # --------------------------------------------------------
    # Explicit JSON request parsing.
    # This prevents FastAPI's default 422 response.
    # --------------------------------------------------------

    try:
        payload = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(payload, dict):
        return invalid_input()

    policy = payload.get("policy")
    objects = payload.get("objects")

    if not isinstance(policy, dict):
        return invalid_input()

    if not isinstance(objects, list):
        return invalid_input()

    # ========================================================
    # Policy
    # ========================================================

    min_result = parse_timestamp(
        policy.get("minTime")
    )

    max_result = parse_timestamp(
        policy.get("maxTime")
    )

    threshold = policy.get(
        "contaminationThreshold"
    )

    policy_invalid = False

    if min_result is None:
        policy_invalid = True

    if max_result is None:
        policy_invalid = True

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
        or threshold > 1
    ):
        policy_invalid = True

    if not policy_invalid:
        min_time = min_result[0]
        max_time = max_result[0]

        if min_time > max_time:
            policy_invalid = True

    # ========================================================
    # Storage
    # ========================================================

    rejected_objects = []
    rejected_rows = []
    lineage = []

    candidates = []

    # ========================================================
    # Object validation
    # ========================================================

    for obj in objects:

        # A non-object cannot provide a usable URI.
        if not isinstance(obj, dict):

            rejected_objects.append({
                "uri": None,
                "reasonCodes": sorted_codes([
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ]),
            })

            continue

        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_generation = obj.get(
            "fetchedGeneration"
        )
        supplied_crc = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

        object_codes = []

        # ----------------------------------------------------
        # URI
        # ----------------------------------------------------

        if (
            not isinstance(uri, str)
            or URI_RE.fullmatch(uri) is None
        ):
            add_code(
                object_codes,
                "URI_INVALID",
            )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        generation_ok = valid_generation(
            generation
        )

        fetched_generation_ok = valid_generation(
            fetched_generation
        )

        if not generation_ok or not fetched_generation_ok:
            add_code(
                object_codes,
                "GENERATION_INVALID",
            )

        # Mismatch is based on unequal supplied values.
        if (
            isinstance(generation, str)
            and isinstance(fetched_generation, str)
            and generation != fetched_generation
        ):
            add_code(
                object_codes,
                "GENERATION_MISMATCH",
            )

        # ----------------------------------------------------
        # CRC syntax
        # ----------------------------------------------------

        crc_ok = valid_crc_syntax(
            supplied_crc
        )

        if not crc_ok:
            add_code(
                object_codes,
                "CRC32C_INVALID",
            )

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        if not isinstance(content, str):
            add_code(
                object_codes,
                "SCHEMA_INVALID",
            )

        if schema_id != "training-v1":
            add_code(
                object_codes,
                "SCHEMA_INVALID",
            )

        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        parsed_rows = []

        if isinstance(content, str):

            lines = content.splitlines()

            nonblank = [
                line
                for line in lines
                if line.strip()
            ]

            if not nonblank:

                add_code(
                    object_codes,
                    "SCHEMA_INVALID",
                )

            else:

                for line in nonblank:

                    try:
                        row = json.loads(line)
                    except Exception:

                        add_code(
                            object_codes,
                            "JSONL_INVALID",
                        )

                        continue

                    if not isinstance(row, dict):

                        add_code(
                            object_codes,
                            "SCHEMA_INVALID",
                        )

                        continue

                    expected_keys = {
                        "id",
                        "entity",
                        "eventTime",
                        "revision",
                        "text",
                    }

                    # Exact shape means exactly these keys.
                    if set(row.keys()) != expected_keys:

                        add_code(
                            object_codes,
                            "SCHEMA_INVALID",
                        )

                        continue

                    if (
                        not isinstance(row["id"], str)
                        or not isinstance(
                            row["entity"],
                            str,
                        )
                        or not isinstance(
                            row["eventTime"],
                            str,
                        )
                        or not isinstance(
                            row["text"],
                            str,
                        )
                        or not valid_revision(
                            row["revision"]
                        )
                    ):

                        add_code(
                            object_codes,
                            "SCHEMA_INVALID",
                        )

                        continue

                    timestamp = parse_timestamp(
                        row["eventTime"]
                    )

                    if timestamp is None:

                        add_code(
                            object_codes,
                            "SCHEMA_INVALID",
                        )

                        continue

                    parsed_rows.append({
                        "id": row["id"],
                        "entity": row["entity"],
                        "eventTime": timestamp[0],
                        "eventTimeCanonical": timestamp[1],
                        "revision": row["revision"],
                        "text": row["text"],
                    })

        # ----------------------------------------------------
        # CRC mismatch
        # ----------------------------------------------------

        # Only check mismatch when:
        # - content is a string
        # - CRC syntax is valid
        if (
            isinstance(content, str)
            and crc_ok
        ):

            actual_crc = (
                f"{crc32c(content.encode('utf-8')):08x}"
            )

            if actual_crc != supplied_crc:

                add_code(
                    object_codes,
                    "CRC32C_MISMATCH",
                )

        # ----------------------------------------------------
        # Reject entire object if ANY object code exists.
        # ----------------------------------------------------

        if object_codes:

            rejected_objects.append({
                "uri": (
                    uri
                    if isinstance(uri, str)
                    else None
                ),
                "reasonCodes": sorted_codes(
                    object_codes
                ),
            })

            continue

        # ====================================================
        # Valid object → lineage
        # ====================================================

        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": supplied_crc,
            "schemaId": schema_id,
        })

        # ====================================================
        # Candidate rows
        # ====================================================

        for row in parsed_rows:

            entity = canonicalize(
                row["entity"]
            )

            text = canonicalize(
                row["text"]
            )

            dedup_key = (
                entity,
                row["eventTimeCanonical"],
                text,
            )

            candidates.append({
                "id": row["id"],
                "entity": entity,
                "eventTime": row["eventTime"],
                "eventTimeCanonical": (
                    row["eventTimeCanonical"]
                ),
                "revision": row["revision"],
                "text": text,
                "dedupKey": dedup_key,
            })

    # ========================================================
    # Deduplication
    # ========================================================

    groups = {}

    for row in candidates:
        groups.setdefault(
            row["dedupKey"],
            [],
        ).append(row)

    retained = []

    for rows in groups.values():

        highest_revision = max(
            row["revision"]
            for row in rows
        )

        highest = [
            row
            for row in rows
            if row["revision"]
            == highest_revision
        ]

        winner = min(
            highest,
            key=lambda row: utf8(row["id"]),
        )

        retained.append(winner)

        for row in rows:

            if row is winner:
                continue

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "DUPLICATE"
                ],
            })

    # ========================================================
    # Policy / time window
    # ========================================================

    usable_rows = []

    for row in retained:

        codes = []

        if policy_invalid:

            add_code(
                codes,
                "POLICY_INVALID",
            )

        else:

            if (
                row["eventTime"] < min_time
                or row["eventTime"] > max_time
            ):
                add_code(
                    codes,
                    "OUT_OF_WINDOW",
                )

        if codes:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": sorted_codes(
                    codes
                ),
            })

        else:
            usable_rows.append(row)

    # ========================================================
    # Deterministic split
    # ========================================================

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in usable_rows:

        split = split_name(
            bucket(row["entity"])
        )

        split_rows[split].append(row)

    # ========================================================
    # Contamination
    # ========================================================

    train_word_sets = [
        word_set(row["text"])
        for row in split_rows["train"]
    ]

    for split in ("validation", "test"):

        kept = []

        for row in split_rows[split]:

            candidate_words = word_set(
                row["text"]
            )

            contaminated = any(
                jaccard(
                    candidate_words,
                    train_words,
                ) >= threshold
                for train_words in train_word_sets
            )

            if contaminated:

                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": [
                        "TRAIN_CONTAMINATION"
                    ],
                })

            else:
                kept.append(row)

        split_rows[split] = kept

    # ========================================================
    # Final canonical split rows
    # ========================================================

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split in (
        "train",
        "validation",
        "test",
    ):

        for row in split_rows[split]:

            splits[split].append({
                "id": row["id"],
                "entity": row["entity"],
                "eventTime": row[
                    "eventTimeCanonical"
                ],
                "revision": row["revision"],
                "text": row["text"],
            })

        splits[split].sort(
            key=lambda row: (
                utf8(row["id"]),
                utf8(compact_json(row)),
            )
        )

    # ========================================================
    # Digests
    # ========================================================

    digests = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        serialized = "".join(
            compact_json(row) + "\n"
            for row in splits[split]
        ).encode("utf-8")

        digests[split] = hashlib.sha256(
            serialized
        ).hexdigest()

    # ========================================================
    # Merge rejected rows with same ID
    # ========================================================

    rejected_by_id = {}

    for item in rejected_rows:

        row_id = item["id"]

        rejected_by_id.setdefault(
            row_id,
            set(),
        ).update(
            item["reasonCodes"]
        )

    rejected_rows = [
        {
            "id": row_id,
            "reasonCodes": sorted_codes(
                list(codes)
            ),
        }
        for row_id, codes
        in rejected_by_id.items()
    ]

    rejected_rows = sort_rejected_rows(
        rejected_rows
    )

    # ========================================================
    # Final deterministic ordering
    # ========================================================

    rejected_objects = sort_rejected_objects(
        rejected_objects
    )

    lineage = sort_lineage(lineage)

    # ========================================================
    # Exact response shape
    # ========================================================

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok"}