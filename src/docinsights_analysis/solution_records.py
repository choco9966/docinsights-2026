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
from pathlib import Path
from typing import Any

from .blind_review import _normalized_answer

_RECORD_FIELDS = frozenset(
    {
        "instance_id",
        "split",
        "question",
        "solution",
        "answer",
        "evidence",
        "evidence_details",
        "source_pages",
        "provenance",
        "uncertainties",
    }
)
_TASK_FIELDS = frozenset({"instance_id", "user_query", "document_pdf"})
_PAGE_FIELDS = frozenset({"page_number", "text"})
_DETAIL_FIELDS = frozenset({"id", "page", "quote"})
_CALCULATION_FIELDS = frozenset({"expression", "result"})
_SOLUTION_FIELDS = frozenset({"summary", "calculations"})
_BLOCK_HEADING = re.compile(r"(?m)^[ \t]*(\S+)[ \t]*:")
_DECIMAL_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_ARITHMETIC_TEXT = re.compile(r"^[0-9eE.+\-*/%() \t]+$")
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 64
_MAX_ABS_VALUE = Decimal("1e100")
_MAX_SIGNIFICANT_DIGITS = 50
_MAX_DECIMAL_EXPONENT = 100


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

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise SolutionRecordError("invalid arithmetic expression")
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise SolutionRecordError("invalid arithmetic expression")
            literal = ast.get_source_segment(expression, node)
            if literal is None or _DECIMAL_TEXT.fullmatch(literal) is None:
                raise SolutionRecordError("invalid arithmetic expression")
            return checked(_decimal_text(literal, "numeric literal"))
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

    return evaluate(tree)


def _ocr_blocks(page_text: str) -> list[dict[str, Any]]:
    # A standalone non-whitespace token followed by a colon is a visible block marker.
    # Body lines with the same shape are conservatively treated as boundaries, which can
    # produce a false negative but prevents a quote from leaking in from an uncited block.
    matches = list(_BLOCK_HEADING.finditer(page_text))
    blocks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
        blocks.append(
            {
                "id": match.group(1),
                "body": page_text[match.end() : end],
                "heading_line_index": page_text.count("\n", 0, match.start()),
            }
        )
    return blocks


def _block_quotes(page_text: str, evidence_id: str) -> list[str]:
    return [block["body"] for block in _ocr_blocks(page_text) if block["id"] == evidence_id]


def visual_evidence_candidates(
    record: dict[str, Any], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Find exact quote-bound candidates whose solver ID is absent from OCR headings."""
    if not isinstance(record, dict):
        raise SolutionRecordError("record must be an object")
    normalized_pages = _normalize_pages(pages)
    evidence = record.get("evidence")
    details = record.get("evidence_details")
    if not isinstance(evidence, list) or any(
        not isinstance(value, str) or not value for value in evidence
    ):
        raise SolutionRecordError("evidence IDs must be non-empty strings")
    if not isinstance(details, list):
        raise SolutionRecordError("evidence_details must be a list")

    blocks_by_page = {
        page["page_number"]: _ocr_blocks(page["text"]) for page in normalized_pages
    }
    all_ocr_ids = {
        block["id"] for blocks in blocks_by_page.values() for block in blocks
    }
    candidates: list[dict[str, Any]] = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise SolutionRecordError(f"evidence_details[{index}] must be an object")
        _require_fields(detail, _DETAIL_FIELDS, f"evidence_details[{index}]")
        evidence_id = _nonempty_string(detail["id"], f"evidence_details[{index}].id")
        quote = _nonempty_string(detail["quote"], f"evidence_details[{index}].quote")
        page_number = detail["page"]
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise SolutionRecordError(f"evidence_details[{index}].page must be an integer")
        if evidence_id in all_ocr_ids or page_number not in blocks_by_page:
            continue
        matching_blocks = [
            block for block in blocks_by_page[page_number] if quote in block["body"]
        ]
        if len(matching_blocks) != 1:
            continue
        block = matching_blocks[0]
        candidates.append(
            {
                "id": evidence_id,
                "ocr_id": block["id"],
                "page": page_number,
                "quote": quote,
                "heading_line_index": block["heading_line_index"],
            }
        )
    return candidates


def _validated_visual_checks(
    record: dict[str, Any],
    pages: list[dict[str, Any]],
    provenance: dict[str, Any],
    supplied: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    recorded = provenance.get("visual_id_checks")
    if recorded is not None and supplied is None:
        raise SolutionRecordError("visual ID checks must be independently supplied")
    if supplied is None:
        return {}
    if not isinstance(supplied, list) or recorded != supplied:
        raise SolutionRecordError(
            "independently supplied visual ID checks must exactly match provenance"
        )

    candidates_by_id = {
        candidate["id"]: candidate for candidate in visual_evidence_candidates(record, pages)
    }
    by_id: dict[str, dict[str, Any]] = {}
    required = {
        "id",
        "visible_id",
        "ocr_id",
        "page",
        "heading_line_index",
        "quote_sha256",
        "page_image_sha256",
        "crop_sha256",
        "clear",
    }

    def validate_hashes(value: Any, field: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{field}.{key}"
                if (
                    isinstance(key, str)
                    and (key == "sha256" or key.endswith("_sha256"))
                    and (not isinstance(item, str) or _SHA256_HEX.fullmatch(item) is None)
                ):
                    raise SolutionRecordError(
                        f"{child} must be a 64-character SHA-256"
                    )
                validate_hashes(item, child)
        elif isinstance(value, list):
            for item_index, item in enumerate(value):
                validate_hashes(item, f"{field}[{item_index}]")

    for index, proof in enumerate(supplied):
        if not isinstance(proof, dict) or not required.issubset(proof):
            raise SolutionRecordError(f"visual ID check {index} is incomplete")
        try:
            json.dumps(proof, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SolutionRecordError(f"visual ID check {index} must be JSON-safe") from error
        proof_id = proof.get("id")
        if not isinstance(proof_id, str):
            raise SolutionRecordError(f"visual ID check {index}.id must be a string")
        candidate = candidates_by_id.get(proof_id)
        if candidate is None or proof_id in by_id:
            raise SolutionRecordError(
                f"visual ID check {index} has no unique evidence block candidate"
            )
        if proof.get("clear") is not True or proof.get("visible_id") != proof_id:
            raise SolutionRecordError(f"visual ID check {index} is not clear and exact")
        if (
            not isinstance(proof.get("ocr_id"), str)
            or isinstance(proof.get("page"), bool)
            or not isinstance(proof.get("page"), int)
            or isinstance(proof.get("heading_line_index"), bool)
            or not isinstance(proof.get("heading_line_index"), int)
        ):
            raise SolutionRecordError(f"visual ID check {index} has invalid binding types")
        if any(
            proof.get(field) != candidate[field]
            for field in ("ocr_id", "page", "heading_line_index")
        ):
            raise SolutionRecordError(f"visual ID check {index} does not bind its OCR block")
        expected_quote_hash = hashlib.sha256(candidate["quote"].encode("utf-8")).hexdigest()
        if proof.get("quote_sha256") != expected_quote_hash:
            raise SolutionRecordError(f"visual ID check {index} does not bind its quote")
        validate_hashes(proof, f"visual ID check {index}")
        by_id[proof_id] = deepcopy(proof)
    return by_id


def validate_solution(
    record: dict[str, Any],
    task: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    visual_id_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one solution solely against its task and extracted public pages."""
    if not isinstance(record, dict):
        raise SolutionRecordError("record must be an object")
    if not isinstance(task, dict):
        raise SolutionRecordError("task must be an object")
    _require_fields(record, _RECORD_FIELDS, "record")
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

    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        raise SolutionRecordError("provenance must be an object")
    visual_checks_by_evidence_id = _validated_visual_checks(
        record, normalized_pages, provenance, visual_id_checks
    )

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
    numeric_answer = _DECIMAL_TEXT.fullmatch(answer) is not None
    answer_value = _decimal_text(answer, "answer") if numeric_answer else None
    if normalized_calculations:
        if answer_value is None:
            raise SolutionRecordError(
                "answer must be numeric when calculations are present"
            )
        last_result = _decimal_text(
            normalized_calculations[-1]["result"], "last calculation result"
        )
        if answer_value != last_result:
            raise SolutionRecordError("final answer does not match the last calculation result")

    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise SolutionRecordError("evidence must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in evidence):
        raise SolutionRecordError("evidence IDs must be non-empty strings")
    if len(evidence) != len(set(evidence)):
        raise SolutionRecordError("duplicate evidence ID")

    details = record["evidence_details"]
    if not isinstance(details, list):
        raise SolutionRecordError("evidence_details must be a list")
    normalized_details: list[dict[str, Any]] = []
    detail_ids: list[str] = []
    page_index = {page["page_number"]: page["text"] for page in normalized_pages}
    for index, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise SolutionRecordError(f"evidence_details[{index}] must be an object")
        _require_fields(detail, _DETAIL_FIELDS, f"evidence_details[{index}]")
        evidence_id = _nonempty_string(detail["id"], f"evidence_details[{index}].id")
        page_number = detail["page"]
        quote = _nonempty_string(detail["quote"], f"evidence_details[{index}].quote")
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise SolutionRecordError(f"evidence_details[{index}].page must be an integer")
        if page_number not in page_index:
            raise SolutionRecordError(
                f"evidence_details[{index}].page is not present in source_pages"
            )
        blocks = _block_quotes(page_index[page_number], evidence_id)
        if not any(quote in block for block in blocks) and (
            evidence_id not in visual_checks_by_evidence_id
        ):
            raise SolutionRecordError(
                f"quote does not belong to evidence block {evidence_id!r} on page {page_number}"
            )
        detail_ids.append(evidence_id)
        normalized_details.append({"id": evidence_id, "page": page_number, "quote": quote})
    if len(detail_ids) != len(set(detail_ids)) or detail_ids != evidence:
        raise SolutionRecordError("evidence_details IDs must uniquely and in order match evidence")

    required_provenance = {
        "pdf_sha256",
        "input_sha256",
        "config_sha256",
        "model",
        "method",
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    if missing_provenance:
        raise SolutionRecordError(
            f"provenance missing fields: {', '.join(missing_provenance)}"
        )
    normalized_provenance = deepcopy(provenance)
    for field in ("pdf_sha256", "input_sha256", "config_sha256"):
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
    if answer_value is None and not normalized_uncertainties:
        raise SolutionRecordError(
            "a nonnumeric answer requires an explicit uncertainty"
        )

    return {
        "instance_id": instance_id,
        "split": split,
        "question": question,
        "solution": {"summary": summary, "calculations": normalized_calculations},
        "answer": answer,
        "evidence": list(evidence),
        "evidence_details": normalized_details,
        "source_pages": deepcopy(normalized_pages),
        "provenance": normalized_provenance,
        "uncertainties": normalized_uncertainties,
    }


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
            ["### 정답 (Answer)\n", f"{record['answer']}\n", "### Evidence\n"]
        )
        for detail in record["evidence_details"]:
            sections.append(f"- `{detail['id']}` (page {detail['page']}): {detail['quote']}\n")
        sections.append("### 불확실성 (Uncertainties)\n")
        sections.append(
            "\n".join(f"- {item}" for item in record["uncertainties"])
            + ("\n" if record["uncertainties"] else "None.\n")
        )
    return "\n".join(sections)


def export_records(
    records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    output_dir: Path | str,
    *,
    source_pages_by_id: Mapping[str, list[dict[str, Any]]],
    expected_split: str,
    visual_checks_by_id: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate complete task coverage and write a private, reviewable export."""
    split = _nonempty_string(expected_split, "expected_split")
    if split not in {"heldout", "train", "validation"}:
        raise SolutionRecordError("expected_split must be heldout, train, or validation")
    if not isinstance(source_pages_by_id, Mapping):
        raise SolutionRecordError("source_pages_by_id must be a mapping")
    if visual_checks_by_id is not None and not isinstance(visual_checks_by_id, Mapping):
        raise SolutionRecordError("visual_checks_by_id must be a mapping")
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
    visual_check_ids = set(visual_checks_by_id or {})
    if any(not isinstance(instance_id, str) for instance_id in visual_check_ids):
        raise SolutionRecordError("visual check instance IDs must be strings")
    unknown_visual_checks = sorted(visual_check_ids - task_ids)
    if unknown_visual_checks:
        raise SolutionRecordError(
            f"unknown visual checks instance_id: {', '.join(unknown_visual_checks)}"
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
            visual_id_checks=(visual_checks_by_id or {}).get(instance_id),
        )
        for instance_id in sorted(task_ids)
    ]
    submissions = [
        {
            "instance_id": record["instance_id"],
            "answer": record["answer"],
            "evidence": record["evidence"],
        }
        for record in normalized
    ]
    content = {
        "solutions.jsonl": _jsonl(normalized),
        "solutions.md": _markdown(normalized),
        "submission.jsonl": _jsonl(submissions),
    }
    destination = ensure_private_directory(output_dir)
    for name in (*content, "manifest.json"):
        _reject_symlink_components(destination / name)
    for name, text in content.items():
        write_private_atomic(destination / name, text)

    manifest = {
        "schema_version": 1,
        "private": True,
        "split": split,
        "total": len(normalized),
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
        record_answer, record_evidence = _comparison_values(record, "record", instance_id)
        reference_answer, reference_evidence = _comparison_values(
            reference, "reference", instance_id
        )
        answer_match = _normalized_answer(record_answer) == _normalized_answer(reference_answer)
        evidence_match = set(record_evidence) == set(reference_evidence)
        comparisons.append(
            {
                "instance_id": instance_id,
                "answer_match": answer_match,
                "evidence_match": evidence_match,
            }
        )
    return {
        "comparisons": comparisons,
        "aggregate": {
            "reference_kind": kind,
            "total": len(comparisons),
            "answer_matches": sum(row["answer_match"] for row in comparisons),
            "evidence_matches": sum(row["evidence_match"] for row in comparisons),
        },
    }


def _comparison_values(
    row: dict[str, Any], kind: str, instance_id: str
) -> tuple[str, list[str]]:
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise SolutionRecordError(f"{kind} {instance_id} answer must be a non-empty string")
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
    return answer, evidence
