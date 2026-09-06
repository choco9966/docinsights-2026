"""Validation and private export helpers for auditable solution records."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from decimal import Decimal, Inexact, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from .blind_review import _normalized_answer

_PRIMARY_FIELDS = frozenset(
    {
        "instance_id",
        "split",
        "question",
        "solution",
        "answer",
        "evidence_regions",
        "source_pages",
        "provenance",
        "uncertainties",
    }
)
_RECORD_FIELDS = _PRIMARY_FIELDS | frozenset(
    {"source_status", "evidence", "evidence_details", "source_uncertainties"}
)
_TASK_FIELDS = frozenset({"instance_id", "user_query", "document_pdf"})
_PAGE_FIELDS = frozenset({"page_number", "text"})
_DETAIL_FIELDS = frozenset({"id", "page", "quote"})
_REGION_FIELDS = frozenset({"page", "ocr_anchor"})
_CALCULATION_FIELDS = frozenset({"expression", "result"})
_SOLUTION_FIELDS = frozenset({"summary", "calculations"})
_DECIMAL_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_ARITHMETIC_TEXT = re.compile(r"^[0-9eE.+\-*/%(), \tround]+$")
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_SOURCE_CHECK_FIELDS = frozenset(
    {
        "region_index",
        "page",
        "ocr_anchor",
        "ocr_anchor_sha256",
        "primary_record_sha256",
        "primary_response_sha256",
        "status",
        "evidence",
        "observed_candidates",
        "source_uncertainties",
        "pdf_sha256",
        "renderer_sha256",
        "checker_executable_sha256",
        "selector_sha256",
        "prompt_sha256",
        "output_schema_sha256",
        "checker_config_sha256",
        "page_image_sha256",
        "ocr_image_sha256",
        "context_crop_sha256",
        "anchor_crop_sha256",
        "raw_response_sha256",
        "model",
        "method",
        "anchor_bbox",
        "context_bbox",
        "anchor_crop_bbox",
        "pdf_artifact",
        "renderer_artifact",
        "page_image_artifact",
        "context_crop_artifact",
        "anchor_crop_artifact",
        "checker_executable_artifact",
        "raw_response_artifact",
        "artifacts",
    }
)
_OBSERVED_FIELDS = frozenset(
    {
        "heading_line",
        "body_text",
        "heading_legibility",
        "body_legibility",
        "contains_anchor_region",
    }
)
_ARTIFACT_FIELDS = frozenset({"kind", "path", "sha256"})
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 64
_MAX_ABS_VALUE = Decimal("1e100")
_MAX_SIGNIFICANT_DIGITS = 50
_MAX_DECIMAL_EXPONENT = 100
_MAX_RATIONAL_DIGITS = 256
_MAX_ROUND_PLACES = 12


class SolutionRecordError(ValueError):
    """A solution record cannot be verified against its public inputs."""


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            mode = os.lstat(component).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise SolutionRecordError(f"private output path contains a symbolic link: {component}")


def ensure_private_directory(path: Path | str) -> Path:
    """Create a mode-0700 directory without following existing symlink components."""
    destination = _absolute_path(path)
    _reject_symlink_components(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(destination)
    if not destination.is_dir():
        raise SolutionRecordError(f"private output directory is invalid: {destination}")
    destination.chmod(0o700)
    return destination


def write_private_atomic(
    path: Path | str, data: str | bytes, *, overwrite: bool = True
) -> None:
    """Write mode-0600 data through a random same-directory file and atomic publish."""
    target = _absolute_path(path)
    _reject_symlink_components(target)
    parent = target.parent
    if not parent.is_dir():
        raise SolutionRecordError(f"private output parent does not exist: {parent}")
    payload = data.encode("utf-8") if isinstance(data, str) else data
    if not isinstance(payload, bytes):
        raise TypeError("private output data must be str or bytes")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(parent, directory_flags)
    temporary_name: str | None = None
    try:
        try:
            target_stat = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None:
            if stat.S_ISLNK(target_stat.st_mode):
                raise SolutionRecordError(f"private output target is a symbolic link: {target}")
            if not stat.S_ISREG(target_stat.st_mode):
                raise SolutionRecordError(f"private output target is not a regular file: {target}")
            if not overwrite:
                raise FileExistsError(target)

        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        for _ in range(10):
            candidate = f".{target.name}.tmp-{secrets.token_hex(16)}"
            try:
                descriptor = os.open(candidate, file_flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise SolutionRecordError("could not allocate a private temporary output file")
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())

        if overwrite:
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            try:
                os.link(
                    temporary_name,
                    target.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    existing = os.stat(
                        target.name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    raise
                if stat.S_ISLNK(existing.st_mode):
                    raise SolutionRecordError(
                        f"private output target is a symbolic link: {target}"
                    ) from None
                raise
            os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SolutionRecordError(f"{field} must be a non-empty string")
    return value.strip()


def _require_fields(value: dict[str, Any], expected: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise SolutionRecordError(f"{field} missing fields: {', '.join(missing)}")
    if extra:
        raise SolutionRecordError(f"{field} has unsupported fields: {', '.join(extra)}")


def _normalize_pages(pages: Any) -> list[dict[str, Any]]:
    if not isinstance(pages, list) or not pages:
        raise SolutionRecordError("pages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise SolutionRecordError(f"pages[{index}] must be an object")
        _require_fields(page, _PAGE_FIELDS, f"pages[{index}]")
        page_number = page["page_number"]
        text = page["text"]
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise SolutionRecordError(f"pages[{index}].page_number must be a positive integer")
        if page_number in seen:
            raise SolutionRecordError(f"duplicate page_number: {page_number}")
        if not isinstance(text, str):
            raise SolutionRecordError(f"pages[{index}].text must be a string")
        seen.add(page_number)
        normalized.append({"page_number": page_number, "text": text})
    return normalized


def input_digest(task: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    """Hash only the official public task fields and independently supplied pages."""
    if not isinstance(task, dict):
        raise SolutionRecordError("task must be an object")
    missing = sorted(_TASK_FIELDS - set(task))
    if missing:
        raise SolutionRecordError(f"task missing fields: {', '.join(missing)}")
    official_task = {
        field: _nonempty_string(task[field], f"task.{field}") for field in sorted(_TASK_FIELDS)
    }
    payload = {"task": official_task, "source_pages": _normalize_pages(pages)}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decimal_text(value: Any, field: str) -> Decimal:
    if not isinstance(value, str) or _DECIMAL_TEXT.fullmatch(value.strip()) is None:
        raise SolutionRecordError(f"{field} must be a numeric string")
    try:
        result = Decimal(value.strip())
    except InvalidOperation as error:
        raise SolutionRecordError(f"{field} must be a numeric string") from error
    _validate_decimal_bound(result, field)
    return result


def _validate_decimal_bound(value: Decimal, field: str) -> None:
    if not value.is_finite() or abs(value) > _MAX_ABS_VALUE:
        raise SolutionRecordError(f"{field} is outside the supported numeric range")
    decimal_tuple = value.as_tuple()
    if len(decimal_tuple.digits) > _MAX_SIGNIFICANT_DIGITS:
        raise SolutionRecordError(
            f"{field} exceeds the supported 50 significant digits"
        )
    if not isinstance(decimal_tuple.exponent, int) or abs(decimal_tuple.exponent) > (
        _MAX_DECIMAL_EXPONENT
    ):
        raise SolutionRecordError(f"{field} has an exponent outside the supported range")


def _evaluate_arithmetic(expression: Any) -> Decimal:
    if (
        not isinstance(expression, str)
        or not expression.strip()
        or len(expression) > _MAX_EXPRESSION_LENGTH
        or _ARITHMETIC_TEXT.fullmatch(expression) is None
    ):
        raise SolutionRecordError("invalid arithmetic expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise SolutionRecordError("invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise SolutionRecordError("arithmetic expression is too complex")

    def checked(value: Decimal) -> Decimal:
        _validate_decimal_bound(value, "arithmetic expression")
        return value

    def decimal_literal(node: ast.Constant) -> Decimal:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SolutionRecordError("invalid arithmetic expression")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise SolutionRecordError("invalid arithmetic expression")
        literal = ast.get_source_segment(expression, node)
        if literal is None or _DECIMAL_TEXT.fullmatch(literal) is None:
            raise SolutionRecordError("invalid arithmetic expression")
        return _decimal_text(literal, "numeric literal")

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return checked(decimal_literal(node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else checked(value.copy_negate())
        if not isinstance(node, ast.BinOp):
            raise SolutionRecordError("invalid arithmetic expression")
        left = evaluate(node.left)
        right = evaluate(node.right)
        try:
            with localcontext() as context:
                context.prec = _MAX_SIGNIFICANT_DIGITS
                context.traps[Inexact] = True
                if isinstance(node.op, ast.Add):
                    return checked(left + right)
                if isinstance(node.op, ast.Sub):
                    return checked(left - right)
                if isinstance(node.op, ast.Mult):
                    return checked(left * right)
                if isinstance(node.op, ast.Div):
                    return checked(left / right)
        except Inexact as error:
            raise SolutionRecordError(
                "arithmetic expression requires unsupported precision"
            ) from error
        except (ArithmeticError, InvalidOperation, OverflowError) as error:
            raise SolutionRecordError("arithmetic expression cannot be evaluated") from error
        raise SolutionRecordError("invalid arithmetic expression")

    def checked_fraction(value: Fraction) -> Fraction:
        if (
            len(str(abs(value.numerator))) > _MAX_RATIONAL_DIGITS
            or len(str(value.denominator)) > _MAX_RATIONAL_DIGITS
            or abs(value) > Fraction(10**_MAX_DECIMAL_EXPONENT)
        ):
            raise SolutionRecordError("arithmetic expression is outside the supported range")
        return value

    def evaluate_fraction(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant):
            return checked_fraction(Fraction(decimal_literal(node)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate_fraction(node.operand)
            return value if isinstance(node.op, ast.UAdd) else checked_fraction(-value)
        if not isinstance(node, ast.BinOp):
            raise SolutionRecordError("invalid arithmetic expression")
        left = evaluate_fraction(node.left)
        right = evaluate_fraction(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return checked_fraction(left + right)
            if isinstance(node.op, ast.Sub):
                return checked_fraction(left - right)
            if isinstance(node.op, ast.Mult):
                return checked_fraction(left * right)
            if isinstance(node.op, ast.Div):
                return checked_fraction(left / right)
        except (ArithmeticError, OverflowError) as error:
            raise SolutionRecordError("arithmetic expression cannot be evaluated") from error
        raise SolutionRecordError("invalid arithmetic expression")

    def evaluate_round(node: ast.Call) -> Decimal:
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "round"
            or node.keywords
            or len(node.args) != 2
        ):
            raise SolutionRecordError("invalid arithmetic expression")
        places_node = node.args[1]
        if (
            not isinstance(places_node, ast.Constant)
            or isinstance(places_node.value, bool)
            or not isinstance(places_node.value, int)
            or not 0 <= places_node.value <= _MAX_ROUND_PLACES
        ):
            raise SolutionRecordError("arithmetic expression has invalid rounding precision")
        value = evaluate_fraction(node.args[0])
        scale = 10**places_node.value
        quotient, remainder = divmod(abs(value.numerator) * scale, value.denominator)
        if remainder * 2 >= value.denominator:
            quotient += 1
        digits = tuple(int(character) for character in str(quotient))
        rounded = Decimal((int(value < 0), digits, -places_node.value))
        return checked(rounded)

    if isinstance(tree.body, ast.Call):
        return evaluate_round(tree.body)
    return evaluate(tree)


def primary_record_digest(primary: Mapping[str, Any]) -> str:
    """Hash an already normalized primary record without reference data."""
    if not isinstance(primary, Mapping) or set(primary) != set(_PRIMARY_FIELDS):
        raise SolutionRecordError("primary record has invalid fields")
    try:
        payload = json.dumps(
            primary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SolutionRecordError("primary record must contain JSON-safe values") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_primary_solution(
    record: dict[str, Any], task: dict[str, Any], pages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate the immutable answer/solution and exact OCR region locators."""
    if not isinstance(record, dict):
        raise SolutionRecordError("record must be an object")
    if not isinstance(task, dict):
        raise SolutionRecordError("task must be an object")
    _require_fields(record, _PRIMARY_FIELDS, "primary record")
    missing_task_fields = sorted(_TASK_FIELDS - set(task))
    if missing_task_fields:
        raise SolutionRecordError(f"task missing fields: {', '.join(missing_task_fields)}")

    normalized_pages = _normalize_pages(pages)
    supplied_source_pages = _normalize_pages(record["source_pages"])
    if supplied_source_pages != normalized_pages:
        raise SolutionRecordError("source_pages do not exactly match the supplied pages")

    instance_id = _nonempty_string(record["instance_id"], "instance_id")
    task_id = _nonempty_string(task["instance_id"], "task.instance_id")
    if instance_id != task_id:
        raise SolutionRecordError("instance_id does not exactly match the task")
    question = _nonempty_string(record["question"], "question")
    _nonempty_string(task["user_query"], "task.user_query")
    if record["question"] != task["user_query"]:
        raise SolutionRecordError("question does not exactly match task.user_query")
    split = _nonempty_string(record["split"], "split")
    if split not in {"heldout", "train", "validation"}:
        raise SolutionRecordError("split must be heldout, train, or validation")

    solution = record["solution"]
    if not isinstance(solution, dict):
        raise SolutionRecordError("solution must be an object")
    _require_fields(solution, _SOLUTION_FIELDS, "solution")
    summary = _nonempty_string(solution["summary"], "solution.summary")
    calculations = solution["calculations"]
    if not isinstance(calculations, list):
        raise SolutionRecordError("solution.calculations must be a list")
    normalized_calculations: list[dict[str, str]] = []
    for index, calculation in enumerate(calculations):
        if not isinstance(calculation, dict):
            raise SolutionRecordError(f"solution.calculations[{index}] must be an object")
        _require_fields(calculation, _CALCULATION_FIELDS, f"solution.calculations[{index}]")
        expression = _nonempty_string(
            calculation["expression"], f"solution.calculations[{index}].expression"
        )
        declared = _decimal_text(
            calculation["result"], f"solution.calculations[{index}].result"
        )
        actual = _evaluate_arithmetic(expression)
        if actual != declared:
            raise SolutionRecordError(
                f"calculation result mismatch at solution.calculations[{index}]"
            )
        normalized_calculations.append(
            {"expression": expression, "result": calculation["result"].strip()}
        )

    answer = _nonempty_string(record["answer"], "answer")
    answer_value = _decimal_text(answer, "answer")
    if normalized_calculations:
        last_result = _decimal_text(
            normalized_calculations[-1]["result"], "last calculation result"
        )
        if answer_value != last_result:
            raise SolutionRecordError("final answer does not match the last calculation result")

    regions = record["evidence_regions"]
    if not isinstance(regions, list) or not regions:
        raise SolutionRecordError("evidence_regions must be a non-empty list")
    normalized_regions: list[dict[str, Any]] = []
    seen_regions: set[tuple[int, str]] = set()
    page_index = {page["page_number"]: page["text"] for page in normalized_pages}
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise SolutionRecordError(f"evidence_regions[{index}] must be an object")
        _require_fields(region, _REGION_FIELDS, f"evidence_regions[{index}]")
        page_number = region["page"]
        anchor = _nonempty_string(region["ocr_anchor"], f"evidence_regions[{index}].ocr_anchor")
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise SolutionRecordError(f"evidence_regions[{index}].page must be an integer")
        if page_number not in page_index:
            raise SolutionRecordError(f"evidence_regions[{index}].page is not present")
        if page_index[page_number].count(anchor) != 1:
            raise SolutionRecordError(
                f"evidence_regions[{index}].ocr_anchor must occur exactly once on its page"
            )
        region_key = (page_number, anchor)
        if region_key in seen_regions:
            raise SolutionRecordError("evidence_regions must not duplicate a locator")
        seen_regions.add(region_key)
        normalized_regions.append({"page": page_number, "ocr_anchor": anchor})

    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        raise SolutionRecordError("provenance must be an object")
    required_provenance = {
        "pdf_sha256",
        "input_sha256",
        "config_sha256",
        "output_sha256",
        "model",
        "method",
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    if missing_provenance:
        raise SolutionRecordError(
            f"provenance missing fields: {', '.join(missing_provenance)}"
        )
    normalized_provenance = deepcopy(provenance)
    for field in ("pdf_sha256", "input_sha256", "config_sha256", "output_sha256"):
        value = provenance[field]
        if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
            raise SolutionRecordError(f"provenance.{field} must be a 64-character SHA-256")
        normalized_provenance[field] = value.casefold()
    if normalized_provenance["input_sha256"] != input_digest(task, normalized_pages):
        raise SolutionRecordError("provenance.input_sha256 does not match the public input")
    for field in ("model", "method"):
        normalized_provenance[field] = _nonempty_string(
            provenance[field], f"provenance.{field}"
        )
    try:
        json.dumps(normalized_provenance, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SolutionRecordError("provenance must contain JSON-safe values") from error
    uncertainties = record["uncertainties"]
    if not isinstance(uncertainties, list) or any(
        not isinstance(value, str) or not value.strip() for value in uncertainties
    ):
        raise SolutionRecordError("uncertainties must be a list of non-empty strings")
    normalized_uncertainties = [value.strip() for value in uncertainties]
    if len(normalized_uncertainties) != len(set(normalized_uncertainties)):
        raise SolutionRecordError("uncertainties must not contain duplicates")
    return {
        "instance_id": instance_id,
        "split": split,
        "question": question,
        "solution": {"summary": summary, "calculations": normalized_calculations},
        "answer": answer,
        "evidence_regions": normalized_regions,
        "source_pages": deepcopy(normalized_pages),
        "provenance": normalized_provenance,
        "uncertainties": normalized_uncertainties,
    }


def _heading_id(heading_line: str) -> str | None:
    for index, character in enumerate(heading_line):
        if character == ":" and index > 0 and (
            index + 1 == len(heading_line) or heading_line[index + 1].isspace()
        ):
            candidate = heading_line[:index]
            if candidate and not any(character.isspace() for character in candidate):
                return candidate
    return None


def _validate_artifact(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SolutionRecordError(f"{field} must be an artifact object")
    _require_fields(value, _ARTIFACT_FIELDS, field)
    kind = _nonempty_string(value["kind"], f"{field}.kind")
    path = _nonempty_string(value["path"], f"{field}.path")
    digest = value["sha256"]
    if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
        raise SolutionRecordError(f"{field}.sha256 must be a 64-character SHA-256")
    return {"kind": kind, "path": path, "sha256": digest.casefold()}


def _validate_bbox(value: Any, field: str) -> None:
    expected = frozenset({"left", "top", "right", "bottom"})
    if not isinstance(value, dict):
        raise SolutionRecordError(f"{field} must be a bbox object")
    _require_fields(value, expected, field)
    left, top, right, bottom = (value[key] for key in ("left", "top", "right", "bottom"))
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in value.values())
        or left < 0
        or top < 0
        or right <= left
        or bottom <= top
    ):
        raise SolutionRecordError(f"{field} has invalid coordinates")


def _normalize_source_checks(
    primary: dict[str, Any], source_checks: Any
) -> list[dict[str, Any]]:
    if not isinstance(source_checks, list):
        raise SolutionRecordError("source_checks must be an independently supplied list")
    regions = primary["evidence_regions"]
    if len(source_checks) != len(regions):
        raise SolutionRecordError("source_checks must cover every evidence region")
    primary_sha = primary_record_digest(primary)
    normalized: list[dict[str, Any]] = []
    hash_fields = [field for field in _SOURCE_CHECK_FIELDS if field.endswith("_sha256")]
    artifact_fields = [
        "pdf_artifact",
        "renderer_artifact",
        "page_image_artifact",
        "context_crop_artifact",
        "anchor_crop_artifact",
        "checker_executable_artifact",
        "raw_response_artifact",
    ]
    for index, raw in enumerate(source_checks):
        if not isinstance(raw, dict):
            raise SolutionRecordError(f"source_checks[{index}] must be an object")
        _require_fields(raw, _SOURCE_CHECK_FIELDS, f"source_checks[{index}]")
        region = regions[index]
        expected_anchor_hash = hashlib.sha256(region["ocr_anchor"].encode("utf-8")).hexdigest()
        if (
            raw["region_index"] != index
            or isinstance(raw["region_index"], bool)
            or not isinstance(raw["region_index"], int)
            or raw["page"] != region["page"]
            or isinstance(raw["page"], bool)
            or not isinstance(raw["page"], int)
            or raw["ocr_anchor"] != region["ocr_anchor"]
            or raw["ocr_anchor_sha256"] != expected_anchor_hash
            or raw["primary_record_sha256"] != primary_sha
            or raw["primary_response_sha256"] != primary["provenance"]["output_sha256"]
            or raw["pdf_sha256"] != primary["provenance"]["pdf_sha256"]
        ):
            raise SolutionRecordError(f"source_checks[{index}] does not bind its primary region")
        for field in hash_fields:
            value = raw[field]
            if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
                raise SolutionRecordError(
                    f"source_checks[{index}].{field} must be a 64-character SHA-256"
                )
        for field in ("model", "method"):
            _nonempty_string(raw[field], f"source_checks[{index}].{field}")
        for field in ("anchor_bbox", "context_bbox", "anchor_crop_bbox"):
            _validate_bbox(raw[field], f"source_checks[{index}].{field}")
        normalized_artifacts = {}
        for field in artifact_fields:
            normalized_artifacts[field] = _validate_artifact(
                raw[field], f"source_checks[{index}].{field}"
            )
        artifact_hash_bindings = {
            "pdf_artifact": "pdf_sha256",
            "renderer_artifact": "renderer_sha256",
            "page_image_artifact": "page_image_sha256",
            "context_crop_artifact": "context_crop_sha256",
            "anchor_crop_artifact": "anchor_crop_sha256",
            "checker_executable_artifact": "checker_executable_sha256",
            "raw_response_artifact": "raw_response_sha256",
        }
        if any(
            normalized_artifacts[artifact]["sha256"] != raw[digest]
            for artifact, digest in artifact_hash_bindings.items()
        ):
            raise SolutionRecordError(
                f"source_checks[{index}] artifact hashes do not match bound hashes"
            )
        if not isinstance(raw["artifacts"], list):
            raise SolutionRecordError(f"source_checks[{index}].artifacts must be a list")
        for artifact_index, artifact in enumerate(raw["artifacts"]):
            _validate_artifact(artifact, f"source_checks[{index}].artifacts[{artifact_index}]")

        observations = raw["observed_candidates"]
        if not isinstance(observations, list):
            raise SolutionRecordError(f"source_checks[{index}].observed_candidates must be a list")
        for observation_index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                raise SolutionRecordError("observed candidate must be an object")
            _require_fields(
                observation,
                _OBSERVED_FIELDS,
                f"source_checks[{index}].observed_candidates[{observation_index}]",
            )
            for field in ("heading_line", "body_text"):
                if observation[field] is not None and not isinstance(observation[field], str):
                    raise SolutionRecordError(f"observed candidate {field} must be text or null")
            for field in ("heading_legibility", "body_legibility"):
                if observation[field] not in {"clear", "ambiguous", "unreadable"}:
                    raise SolutionRecordError(f"observed candidate {field} is invalid")
            if not isinstance(observation["contains_anchor_region"], bool):
                raise SolutionRecordError("observed candidate anchor flag must be boolean")

        uncertainty = raw["source_uncertainties"]
        if not isinstance(uncertainty, list) or any(
            not isinstance(item, str) or not item.strip() for item in uncertainty
        ):
            raise SolutionRecordError(f"source_checks[{index}].source_uncertainties is invalid")
        evidence = raw["evidence"]
        if not isinstance(evidence, dict):
            raise SolutionRecordError(f"source_checks[{index}].evidence must be an object")
        _require_fields(evidence, _DETAIL_FIELDS, f"source_checks[{index}].evidence")
        if (
            evidence["page"] != region["page"]
            or isinstance(evidence["page"], bool)
            or not isinstance(evidence["page"], int)
        ):
            raise SolutionRecordError(f"source_checks[{index}].evidence has wrong page")
        status = raw["status"]
        anchored = [item for item in observations if item["contains_anchor_region"]]
        if status == "fully_grounded":
            evidence_id = _nonempty_string(evidence["id"], "grounded evidence.id")
            quote = _nonempty_string(evidence["quote"], "grounded evidence.quote")
            if uncertainty or len(anchored) != 1:
                raise SolutionRecordError("fully grounded source check is not uniquely clear")
            observed = anchored[0]
            if (
                observed["heading_legibility"] != "clear"
                or observed["body_legibility"] != "clear"
                or _heading_id(observed["heading_line"] or "") != evidence_id
                or observed["body_text"] != quote
            ):
                raise SolutionRecordError("fully grounded evidence does not match pixels")
        elif status == "evidence_unresolved":
            if evidence["id"] is not None or not uncertainty:
                raise SolutionRecordError("unresolved evidence must keep a null ID and uncertainty")
            if evidence["quote"] is not None and (
                not isinstance(evidence["quote"], str)
                or not evidence["quote"].strip()
                or evidence["quote"] not in {
                    item["body_text"] for item in observations if item["body_text"]
                }
            ):
                raise SolutionRecordError("unresolved readable quote is not observed")
        else:
            raise SolutionRecordError("source check status must be terminal")
        normalized.append(deepcopy(raw))
    return normalized


def finalize_solution(
    primary: dict[str, Any],
    task: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    source_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fill final pixel-derived evidence without changing the primary solution."""
    normalized_primary = validate_primary_solution(primary, task, pages)
    checks = _normalize_source_checks(normalized_primary, source_checks)
    evidence_details: list[dict[str, Any]] = []
    grounded_by_id: dict[str, dict[str, Any]] = {}
    for check in checks:
        detail = deepcopy(check["evidence"])
        evidence_id = detail["id"]
        if evidence_id is None:
            evidence_details.append(detail)
            continue
        prior = grounded_by_id.get(evidence_id)
        if prior is not None:
            if prior != detail:
                raise SolutionRecordError(
                    f"grounded evidence ID {evidence_id!r} has conflicting transcriptions"
                )
            continue
        grounded_by_id[evidence_id] = detail
        evidence_details.append(detail)
    source_uncertainties = [
        {"region_index": index, "message": message.strip()}
        for index, check in enumerate(checks)
        for message in check["source_uncertainties"]
    ]
    result = deepcopy(normalized_primary)
    result.update(
        {
            "source_status": (
                "fully_grounded"
                if all(check["status"] == "fully_grounded" for check in checks)
                else "evidence_unresolved"
            ),
            "evidence": [
                detail["id"] for detail in evidence_details if detail["id"] is not None
            ],
            "evidence_details": evidence_details,
            "source_uncertainties": source_uncertainties,
        }
    )
    result["provenance"]["source_checks"] = deepcopy(checks)
    return result


def validate_solution(
    record: dict[str, Any],
    task: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    source_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a final record against independently supplied terminal source checks."""
    if not isinstance(record, dict):
        raise SolutionRecordError("record must be an object")
    _require_fields(record, _RECORD_FIELDS, "record")
    provenance = record.get("provenance")
    recorded_checks = provenance.get("source_checks") if isinstance(provenance, dict) else None
    if source_checks is None or source_checks != recorded_checks:
        raise SolutionRecordError("source_checks must be independently supplied and exact")
    primary = {field: deepcopy(record[field]) for field in _PRIMARY_FIELDS}
    assert isinstance(primary["provenance"], dict)
    primary["provenance"].pop("source_checks", None)
    expected = finalize_solution(primary, task, pages, source_checks=source_checks)
    if record != expected:
        raise SolutionRecordError("final record does not match immutable primary and source checks")
    return expected


def _unique_index(rows: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise SolutionRecordError(f"{kind}s must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SolutionRecordError(f"{kind}[{index}] must be an object")
        instance_id = _nonempty_string(row.get("instance_id"), f"{kind}[{index}].instance_id")
        if instance_id in indexed:
            raise SolutionRecordError(f"duplicate {kind} instance_id: {instance_id}")
        indexed[instance_id] = row
    return indexed


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def _markdown(records: list[dict[str, Any]]) -> str:
    sections = ["# Solution Records\n"]
    for record in records:
        sections.extend(
            [
                f"## {record['instance_id']}\n",
                "### 질문 (Question)\n",
                f"{record['question']}\n",
                "### 풀이 (Solution)\n",
                f"{record['solution']['summary']}\n",
                "### 계산식 (Calculations)\n",
            ]
        )
        calculations = record["solution"]["calculations"]
        sections.append(
            "\n".join(
                f"- `{item['expression']}` = `{item['result']}`" for item in calculations
            )
            + ("\n" if calculations else "None.\n")
        )
        sections.extend(
            [
                "### 정답 (Answer)\n",
                f"{record['answer']}\n",
                "### 근거 상태 (Source Status)\n",
                f"{record['source_status']}\n",
                "### Evidence\n",
            ]
        )
        for detail in record["evidence_details"]:
            evidence_id = detail["id"] if detail["id"] is not None else "unresolved"
            quote = detail["quote"] if detail["quote"] is not None else "unreadable"
            sections.append(f"- `{evidence_id}` (page {detail['page']}): {quote}\n")
        sections.append("### 불확실성 (Uncertainties)\n")
        sections.append(
            "\n".join(f"- {item}" for item in record["uncertainties"])
            + ("\n" if record["uncertainties"] else "None.\n")
        )
        sections.append("### 근거 불확실성 (Source Uncertainties)\n")
        sections.append(
            "\n".join(
                f"- region {item['region_index']}: {item['message']}"
                for item in record["source_uncertainties"]
            )
            + ("\n" if record["source_uncertainties"] else "None.\n")
        )
    return "\n".join(sections)


def export_records(
    records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    output_dir: Path | str,
    *,
    source_pages_by_id: Mapping[str, list[dict[str, Any]]],
    expected_split: str,
    source_checks_by_id: Mapping[str, list[dict[str, Any]]],
    submission_mode: str = "require_fully_grounded",
) -> dict[str, Any]:
    """Validate complete task coverage and write a private, reviewable export."""
    split = _nonempty_string(expected_split, "expected_split")
    if split not in {"heldout", "train", "validation"}:
        raise SolutionRecordError("expected_split must be heldout, train, or validation")
    if submission_mode not in {"require_fully_grounded", "abstain_unresolved"}:
        raise SolutionRecordError("unsupported submission_mode")
    if submission_mode == "abstain_unresolved" and split != "heldout":
        raise SolutionRecordError("abstain_unresolved submission mode is heldout-only")
    if not isinstance(source_pages_by_id, Mapping):
        raise SolutionRecordError("source_pages_by_id must be a mapping")
    if not isinstance(source_checks_by_id, Mapping):
        raise SolutionRecordError("source_checks_by_id must be a mapping")
    record_index = _unique_index(records, "record")
    task_index = _unique_index(tasks, "task")
    record_ids = set(record_index)
    task_ids = set(task_index)
    missing = sorted(task_ids - record_ids)
    unknown = sorted(record_ids - task_ids)
    if missing:
        raise SolutionRecordError(f"missing record instance_id: {', '.join(missing)}")
    if unknown:
        raise SolutionRecordError(f"unknown record instance_id: {', '.join(unknown)}")
    source_ids = set(source_pages_by_id)
    if any(not isinstance(instance_id, str) for instance_id in source_ids):
        raise SolutionRecordError("source page instance IDs must be strings")
    missing_sources = sorted(task_ids - source_ids)
    unknown_sources = sorted(source_ids - task_ids)
    if missing_sources:
        raise SolutionRecordError(
            f"missing source pages instance_id: {', '.join(missing_sources)}"
        )
    if unknown_sources:
        raise SolutionRecordError(
            f"unknown source pages instance_id: {', '.join(unknown_sources)}"
        )
    source_check_ids = set(source_checks_by_id)
    if any(not isinstance(instance_id, str) for instance_id in source_check_ids):
        raise SolutionRecordError("source check instance IDs must be strings")
    missing_source_checks = sorted(task_ids - source_check_ids)
    unknown_source_checks = sorted(source_check_ids - task_ids)
    if missing_source_checks:
        raise SolutionRecordError(
            f"missing source checks instance_id: {', '.join(missing_source_checks)}"
        )
    if unknown_source_checks:
        raise SolutionRecordError(
            f"unknown source checks instance_id: {', '.join(unknown_source_checks)}"
        )
    wrong_splits = sorted(
        instance_id
        for instance_id, record in record_index.items()
        if record.get("split") != split
    )
    if wrong_splits:
        raise SolutionRecordError(
            f"records do not match expected split {split}: {', '.join(wrong_splits)}"
        )

    normalized = [
        validate_solution(
            record_index[instance_id],
            task_index[instance_id],
            source_pages_by_id[instance_id],
            source_checks=source_checks_by_id[instance_id],
        )
        for instance_id in sorted(task_ids)
    ]
    grounded = [record for record in normalized if record["source_status"] == "fully_grounded"]
    unresolved = [
        record for record in normalized if record["source_status"] == "evidence_unresolved"
    ]
    grounded_submissions = [
        {
            "instance_id": record["instance_id"],
            "answer": record["answer"],
            "evidence": record["evidence"],
        }
        for record in grounded
    ]
    submissions: list[dict[str, Any]] | None
    if not unresolved:
        submissions = grounded_submissions
        effective_submission_mode = "fully_grounded"
    elif submission_mode == "abstain_unresolved":
        submissions = [
            (
                {
                    "instance_id": record["instance_id"],
                    "answer": record["answer"],
                    "evidence": record["evidence"],
                }
                if record["source_status"] == "fully_grounded"
                else {"instance_id": record["instance_id"], "answer": None, "evidence": []}
            )
            for record in normalized
        ]
        effective_submission_mode = "heldout_abstentions"
    else:
        submissions = None
        effective_submission_mode = "blocked_unresolved"
    content = {
        "solutions.jsonl": _jsonl(normalized),
        "solutions.md": _markdown(normalized),
        "grounded-submission.jsonl": _jsonl(grounded_submissions),
    }
    if submissions is not None:
        content["submission.jsonl"] = _jsonl(submissions)
    destination = ensure_private_directory(output_dir)
    stale_submission = destination / "submission.jsonl"
    if submissions is None and (stale_submission.exists() or stale_submission.is_symlink()):
        raise SolutionRecordError(
            "blocked export refuses to leave a stale submission.jsonl"
        )
    for name in (*content, "manifest.json"):
        _reject_symlink_components(destination / name)
    for name, text in content.items():
        write_private_atomic(destination / name, text)

    manifest = {
        "schema_version": 2,
        "private": True,
        "split": split,
        "total": len(normalized),
        "answer_coverage": len(normalized),
        "coverage_complete": True,
        "fully_grounded_count": len(grounded),
        "evidence_unresolved_count": len(unresolved),
        "runtime_failed_count": 0,
        "submission_ready": submissions is not None,
        "submission_mode": effective_submission_mode,
        "submission_count": len(submissions) if submissions is not None else 0,
        "grounded_submission_count": len(grounded_submissions),
        "submission_abstention_count": (
            len(unresolved) if effective_submission_mode == "heldout_abstentions" else 0
        ),
        "submission_exclusions": [
            {"instance_id": record["instance_id"], "reason": "evidence_unresolved"}
            for record in unresolved
            if submissions is None
        ],
        "instance_ids": [record["instance_id"] for record in normalized],
        "files": {
            name: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            for name, text in content.items()
        },
    }
    write_private_atomic(
        destination / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def evaluate_records(
    records: list[dict[str, Any]],
    references: list[dict[str, Any]],
    reference_kind: str,
) -> dict[str, Any]:
    """Compare frozen outputs with references without returning solver inputs."""
    kind = _nonempty_string(reference_kind, "reference_kind")
    record_index = _unique_index(records, "record")
    reference_index = _unique_index(references, "reference")
    record_ids = set(record_index)
    reference_ids = set(reference_index)
    missing = sorted(record_ids - reference_ids)
    unknown = sorted(reference_ids - record_ids)
    if missing:
        raise SolutionRecordError(f"missing reference instance_id: {', '.join(missing)}")
    if unknown:
        raise SolutionRecordError(f"unknown reference instance_id: {', '.join(unknown)}")

    comparisons: list[dict[str, Any]] = []
    for instance_id in sorted(record_ids):
        record = record_index[instance_id]
        reference = reference_index[instance_id]
        record_answer = _comparison_answer(record, "record", instance_id)
        reference_answer = _comparison_answer(reference, "reference", instance_id)
        reference_evidence = _comparison_evidence(reference, "reference", instance_id)
        source_status = record.get("source_status")
        if source_status == "fully_grounded":
            record_evidence = _comparison_evidence(record, "record", instance_id)
            evidence_match: bool | None = set(record_evidence) == set(reference_evidence)
            evidence_assessable = True
        elif source_status == "evidence_unresolved":
            evidence_match = None
            evidence_assessable = False
        else:
            raise SolutionRecordError(
                f"record {instance_id} source_status must be terminal"
            )
        answer_match = _normalized_answer(record_answer) == _normalized_answer(reference_answer)
        comparisons.append(
            {
                "instance_id": instance_id,
                "answer_match": answer_match,
                "evidence_assessable": evidence_assessable,
                "evidence_match": evidence_match,
            }
        )
    return {
        "comparisons": comparisons,
        "aggregate": {
            "reference_kind": kind,
            "total": len(comparisons),
            "answer_matches": sum(row["answer_match"] for row in comparisons),
            "evidence_assessable": sum(row["evidence_assessable"] for row in comparisons),
            "evidence_matches": sum(row["evidence_match"] is True for row in comparisons),
            "evidence_unresolved": sum(
                not row["evidence_assessable"] for row in comparisons
            ),
        },
    }


def _comparison_answer(row: dict[str, Any], kind: str, instance_id: str) -> str:
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise SolutionRecordError(f"{kind} {instance_id} answer must be a non-empty string")
    return answer


def _comparison_evidence(
    row: dict[str, Any], kind: str, instance_id: str
) -> list[str]:
    evidence = row.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(value, str) or not value for value in evidence)
        or len(evidence) != len(set(evidence))
    ):
        raise SolutionRecordError(
            f"{kind} {instance_id} evidence must contain unique non-empty strings"
        )
    return evidence
