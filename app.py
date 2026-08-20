import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()


# =========================================================
# Regular expressions
# =========================================================

GENERATION_RE = re.compile(r"^(0|[1-9][0-9]*)$")
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

URI_RE = re.compile(r"^gs://[^/]+/.+$")


# =========================================================
# Basic helpers
# =========================================================

def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def add_reason(reasons: list[str], code: str) -> None:
    if code not in reasons:
        reasons.append(code)


def sorted_reasons(reasons: list[str]) -> list[str]:
    return sorted(set(reasons), key=utf8_key)


# =========================================================
# Unicode canonicalization
# =========================================================

def canonicalize_text(value: str) -> str:
    """
    NFKC -> lowercase -> trim -> collapse Unicode
    whitespace into a single ASCII space.
    """
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # str.split() handles Unicode whitespace.
    return " ".join(value.split())


# =========================================================
# Timestamp validation / canonicalization
# =========================================================

def parse_timestamp(value: Any):
    """
    Accept:
        YYYY-MM-DDTHH:mm:ssZ
        YYYY-MM-DDTHH:mm:ss.sZ
        YYYY-MM-DDTHH:mm:ss.ssZ
        YYYY-MM-DDTHH:mm:ss.sssZ

    or the equivalent numeric UTC offset.

    Returns:
        (UTC datetime, canonical timestamp)
    or:
        None
    """

    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        fraction,
        offset,
    ) = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Validate offset.
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

        # +14:01 and -14:01 are invalid.
        if offset_hour == 14 and offset_minute != 0:
            return None

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute,
            )
        )

    # Calendar and clock validation.
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

    # Convert 1/2/3 digit fraction to milliseconds.
    if fraction is None:
        milliseconds = 0
    elif len(fraction) == 1:
        milliseconds = int(fraction) * 100
    elif len(fraction) == 2:
        milliseconds = int(fraction) * 10
    else:
        milliseconds = int(fraction)

    dt = dt.replace(microsecond=milliseconds * 1000)
    dt = dt.astimezone(timezone.utc)

    canonical = (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{dt.microsecond // 1000:03d}Z"
    )

    return dt, canonical


# =========================================================
# CRC32C / Castagnoli
# =========================================================

def crc32c(data: bytes) -> int:
    """
    CRC32C using the Castagnoli polynomial.
    """

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


def valid_crc_syntax(value: Any) -> bool:
    return (
        isinstance(value, str)
        and CRC32C_RE.fullmatch(value) is not None
    )


# =========================================================
# Generation validation
# =========================================================

def valid_generation(value: Any) -> bool:
    return (
        isinstance(value, str)
        and GENERATION_RE.fullmatch(value) is not None
    )


# =========================================================
# Revision validation
# =========================================================

def valid_revision(value: Any) -> bool:
    """
    Non-negative safe integer.

    JavaScript safe integer maximum:
    2^53 - 1 = 9007199254740991
    """

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


# =========================================================
# Contamination word sets
# =========================================================

def unicode_word_set(value: str) -> set[str]:
    """
    Lowercase Unicode letter/number word set.

    A word consists of consecutive Unicode characters
    whose category starts with L or N.
    """

    result = set()
    current = []

    for char in value.lower():
        category = unicodedata.category(char)

        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        else:
            if current:
                result.add("".join(current))
                current = []

    if current:
        result.add("".join(current))

    return result


def jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:

    if not first and not second:
        return 1.0

    union = first | second

    if not union:
        return 1.0

    return len(first & second) / len(union)


# =========================================================
# Split calculation
# =========================================================

def get_bucket(entity: str) -> int:
    digest = hashlib.sha256(
        entity.encode("utf-8")
    ).digest()

    return digest[0] % 10


def get_split(bucket: int) -> str:
    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


# =========================================================
# Deterministic sorting
# =========================================================

def sort_by_utf8_id(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            utf8_key(row["id"]),
            utf8_key(compact_json(row)),
        ),
    )


def sort_objects(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            utf8_key(row["uri"])
            if isinstance(row["uri"], str)
            else b"",
            utf8_key(compact_json(row)),
        ),
    )


def sort_lineage(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            utf8_key(row["uri"]),
            utf8_key(compact_json(row)),
        ),
    )


# =========================================================
# Invalid input
# =========================================================

def invalid_input_response():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


# =========================================================
# Endpoint
# =========================================================

@app.post("/build-corpus")
async def build_corpus(payload: Any = Body(...)):

    # -----------------------------------------------------
    # Top-level request validation
    # -----------------------------------------------------

    if not isinstance(payload, dict):
        return invalid_input_response()

    policy = payload.get("policy")
    objects = payload.get("objects")

    if not isinstance(policy, dict):
        return invalid_input_response()

    if not isinstance(objects, list):
        return invalid_input_response()

    # -----------------------------------------------------
    # Validate policy
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    rejected_objects = []
    rejected_rows = []
    lineage = []

    candidates = []

    # -----------------------------------------------------
    # Process every object
    # -----------------------------------------------------

    for obj in objects:

        # Non-object supplied.
        if not isinstance(obj, dict):

            rejected_objects.append({
                "uri": None,
                "reasonCodes": [
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ],
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

        object_reasons = []

        # -------------------------------------------------
        # URI
        # -------------------------------------------------

        if (
            not isinstance(uri, str)
            or URI_RE.fullmatch(uri) is None
        ):
            add_reason(
                object_reasons,
                "URI_INVALID",
            )

        # -------------------------------------------------
        # Generations
        # -------------------------------------------------

        generation_valid = valid_generation(
            generation
        )

        fetched_generation_valid = valid_generation(
            fetched_generation
        )

        if (
            not generation_valid
            or not fetched_generation_valid
        ):
            add_reason(
                object_reasons,
                "GENERATION_INVALID",
            )

        if (
            isinstance(generation, str)
            and isinstance(fetched_generation, str)
            and generation != fetched_generation
        ):
            add_reason(
                object_reasons,
                "GENERATION_MISMATCH",
            )

        # -------------------------------------------------
        # CRC syntax
        # -------------------------------------------------

        crc_valid = valid_crc_syntax(
            supplied_crc
        )

        if not crc_valid:
            add_reason(
                object_reasons,
                "CRC32C_INVALID",
            )

        # -------------------------------------------------
        # Schema-level checks
        # -------------------------------------------------

        if not isinstance(content, str):
            add_reason(
                object_reasons,
                "SCHEMA_INVALID",
            )

        if schema_id != "training-v1":
            add_reason(
                object_reasons,
                "SCHEMA_INVALID",
            )

        parsed_rows = []

        # -------------------------------------------------
        # JSONL parsing
        # -------------------------------------------------

        if isinstance(content, str):

            lines = content.splitlines()

            nonblank_exists = any(
                line.strip()
                for line in lines
            )

            if not nonblank_exists:

                add_reason(
                    object_reasons,
                    "SCHEMA_INVALID",
                )

            else:

                for line in lines:

                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except Exception:

                        add_reason(
                            object_reasons,
                            "JSONL_INVALID",
                        )

                        continue

                    if not isinstance(row, dict):

                        add_reason(
                            object_reasons,
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

                    if set(row.keys()) != expected_keys:

                        add_reason(
                            object_reasons,
                            "SCHEMA_INVALID",
                        )

                        continue

                    if (
                        not isinstance(
                            row["id"],
                            str,
                        )
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

                        add_reason(
                            object_reasons,
                            "SCHEMA_INVALID",
                        )

                        continue

                    timestamp_result = parse_timestamp(
                        row["eventTime"]
                    )

                    if timestamp_result is None:

                        add_reason(
                            object_reasons,
                            "SCHEMA_INVALID",
                        )

                        continue

                    parsed_rows.append({
                        "id": row["id"],
                        "entity": row["entity"],
                        "eventTime": timestamp_result[0],
                        "eventTimeCanonical": timestamp_result[1],
                        "revision": row["revision"],
                        "text": row["text"],
                    })

        # -------------------------------------------------
        # CRC content validation
        # -------------------------------------------------

        if (
            isinstance(content, str)
            and crc_valid
        ):

            actual_crc = (
                f"{crc32c(content.encode('utf-8')):08x}"
            )

            if actual_crc != supplied_crc:

                add_reason(
                    object_reasons,
                    "CRC32C_MISMATCH",
                )

        # -------------------------------------------------
        # Reject invalid object
        # -------------------------------------------------

        if object_reasons:

            rejected_objects.append({
                "uri": (
                    uri
                    if isinstance(uri, str)
                    else None
                ),
                "reasonCodes": sorted_reasons(
                    object_reasons
                ),
            })

            continue

        # -------------------------------------------------
        # Valid object lineage
        # -------------------------------------------------

        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": supplied_crc,
            "schemaId": schema_id,
        })

        # -------------------------------------------------
        # Add rows to global candidate set
        # -------------------------------------------------

        for row in parsed_rows:

            canonical_entity = canonicalize_text(
                row["entity"]
            )

            canonical_text = canonicalize_text(
                row["text"]
            )

            key = (
                canonical_entity,
                row["eventTimeCanonical"],
                canonical_text,
            )

            candidates.append({
                "id": row["id"],
                "entity": canonical_entity,
                "eventTime": row["eventTime"],
                "eventTimeCanonical": (
                    row["eventTimeCanonical"]
                ),
                "revision": row["revision"],
                "text": canonical_text,
                "_dedup_key": key,
            })

    # =====================================================
    # Deduplication
    # =====================================================

    grouped = {}

    for row in candidates:

        grouped.setdefault(
            row["_dedup_key"],
            [],
        ).append(row)

    retained = []

    for rows in grouped.values():

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
            key=lambda row: utf8_key(
                row["id"]
            ),
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

    # =====================================================
    # Policy / time window
    # =====================================================

    policy_rows = []

    for row in retained:

        reasons = []

        if policy_invalid:

            add_reason(
                reasons,
                "POLICY_INVALID",
            )

        else:

            if (
                row["eventTime"] < min_time
                or row["eventTime"] > max_time
            ):

                add_reason(
                    reasons,
                    "OUT_OF_WINDOW",
                )

        if reasons:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": sorted_reasons(
                    reasons
                ),
            })

        else:
            policy_rows.append(row)

    # =====================================================
    # Bucket split
    # =====================================================

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in policy_rows:

        bucket = get_bucket(
            row["entity"]
        )

        split_name = get_split(bucket)

        split_rows[split_name].append(row)

    # =====================================================
    # Train contamination
    # =====================================================

    train_sets = [
        unicode_word_set(row["text"])
        for row in split_rows["train"]
    ]

    for split_name in (
        "validation",
        "test",
    ):

        kept = []

        for row in split_rows[split_name]:

            candidate_set = unicode_word_set(
                row["text"]
            )

            contaminated = False

            for train_set in train_sets:

                similarity = jaccard_similarity(
                    candidate_set,
                    train_set,
                )

                if similarity >= threshold:

                    contaminated = True
                    break

            if contaminated:

                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": [
                        "TRAIN_CONTAMINATION"
                    ],
                })

            else:
                kept.append(row)

        split_rows[split_name] = kept

    # =====================================================
    # Canonical output
    # =====================================================

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split_name in (
        "train",
        "validation",
        "test",
    ):

        for row in split_rows[split_name]:

            splits[split_name].append({
                "id": row["id"],
                "entity": row["entity"],
                "eventTime": row[
                    "eventTimeCanonical"
                ],
                "revision": row["revision"],
                "text": row["text"],
            })

        splits[split_name] = sort_by_utf8_id(
            splits[split_name]
        )

    # =====================================================
    # SHA-256 digests
    # =====================================================

    digests = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):

        serialized = "".join(
            compact_json(row) + "\n"
            for row in splits[split_name]
        ).encode("utf-8")

        digests[split_name] = (
            hashlib.sha256(
                serialized
            ).hexdigest()
        )

    # =====================================================
    # Merge rejected rows by ID
    # =====================================================

    rejected_by_id = {}

    for item in rejected_rows:

        row_id = item["id"]

        if row_id not in rejected_by_id:
            rejected_by_id[row_id] = set()

        rejected_by_id[row_id].update(
            item["reasonCodes"]
        )

    rejected_rows_output = []

    for row_id, reasons in rejected_by_id.items():

        rejected_rows_output.append({
            "id": row_id,
            "reasonCodes": sorted_reasons(
                list(reasons)
            ),
        })

    rejected_rows_output.sort(
        key=lambda row: (
            utf8_key(row["id"]),
            utf8_key(compact_json(row)),
        )
    )

    # =====================================================
    # Sort rejected objects
    # =====================================================

    for item in rejected_objects:

        item["reasonCodes"] = sorted_reasons(
            item["reasonCodes"]
        )

    rejected_objects = sort_objects(
        rejected_objects
    )

    # =====================================================
    # Sort lineage
    # =====================================================

    lineage = sort_lineage(lineage)

    # =====================================================
    # Exact response shape
    # =====================================================

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_output,
        "digests": digests,
        "lineage": lineage,
    }


# =========================================================
# Optional health endpoint
# =========================================================

@app.get("/")
async def root():
    return {"status": "ok"}