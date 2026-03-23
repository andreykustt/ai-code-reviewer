from __future__ import annotations

import re

from .evidence_models import code_ref, rule_ref
from .schemas import CandidateFinding, HunkTask, StaticSignal

PY_MUTABLE_DEFAULT = re.compile(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\([^\)]*=\s*(\[\]|\{\}|set\(\))")
PY_BROAD_EXCEPT = re.compile(r"^\s*except\s+Exception\s*:")
PY_SQL_FSTRING = re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE).*(\{.+\}|%s|%\()", re.IGNORECASE)
PY_LOG_SECRET = re.compile(r"(?:logger|logging)\.[A-Za-z_]+\(.*(token|secret|password)", re.IGNORECASE)
PY_AWAITABLE_CALL = re.compile(r"\b(fetch|load|send|notify|publish|request|query|execute|create_task)\w*\(")
PY_TERMINAL_STATEMENT = re.compile(r"^\s*(return\b|raise\b)")


def _line_for_match(task: HunkTask, index: int) -> int:
    if 0 <= index < len(task.changedNewLines):
        return task.changedNewLines[index]
    return task.firstChangedLine


def _matching_rule_ref(task: HunkTask, signals: list[StaticSignal], preferred_type: str | None = None) -> list[str]:
    refs: list[str] = []
    for index, signal in enumerate(signals):
        if preferred_type and signal.type != preferred_type:
            continue
        refs.append(rule_ref(task.taskId, index))
    return refs[:2]


def _line_code_ref(task: HunkTask, index: int) -> str:
    if 0 <= index < len(task.changedNewLines):
        return code_ref(task.taskId, index + 1)
    return code_ref(task.taskId, 0)


def rule_based_bug_candidates(task: HunkTask, signals: list[StaticSignal]) -> list[CandidateFinding]:
    if task.languageSlug != "python":
        return []

    candidates: list[CandidateFinding] = []
    for index, line in enumerate(task.addedLines):
        line_no = _line_for_match(task, index)
        stripped = line.strip()

        if PY_MUTABLE_DEFAULT.search(stripped):
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=line_no,
                    lineEnd=line_no,
                    severity="medium",
                    category="bugs",
                    title="Avoid mutable default arguments",
                    body="This function introduces a mutable default argument, which can retain state across calls and cause unexpected behavior.",
                    confidence=0.92,
                    evidenceRefs=[_line_code_ref(task, index), *_matching_rule_ref(task, signals)],
                )
            )

        if PY_BROAD_EXCEPT.search(stripped):
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=line_no,
                    lineEnd=line_no,
                    severity="medium",
                    category="bugs",
                    title="Avoid broad exception handling",
                    body="Catching Exception directly can hide programming errors and make failures harder to diagnose.",
                    confidence=0.84,
                    evidenceRefs=[_line_code_ref(task, index), *_matching_rule_ref(task, signals)],
                )
            )

        if PY_SQL_FSTRING.search(stripped):
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=line_no,
                    lineEnd=line_no,
                    severity="high",
                    category="security",
                    title="Avoid building SQL queries from interpolated strings",
                    body="This query appears to interpolate runtime values directly into SQL text, which can create injection risk and brittle query behavior.",
                    confidence=0.95,
                    evidenceRefs=[_line_code_ref(task, index), *_matching_rule_ref(task, signals)],
                )
            )

        if PY_LOG_SECRET.search(stripped):
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=line_no,
                    lineEnd=line_no,
                    severity="high",
                    category="security",
                    title="Do not log sensitive authentication data",
                    body="This logging statement appears to include sensitive token or credential data, which should not be written to logs.",
                    confidence=0.9,
                    evidenceRefs=[_line_code_ref(task, index), *_matching_rule_ref(task, signals)],
                )
            )

        if stripped.startswith("return ") and PY_AWAITABLE_CALL.search(stripped) and "await " not in stripped and "asyncio.create_task" not in stripped:
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=line_no,
                    lineEnd=line_no,
                    severity="medium",
                    category="bugs",
                    title="Check whether async work should be awaited",
                    body="This change returns or forwards a call that looks asynchronous without awaiting it. Verify that the surrounding function is intentionally returning the awaitable instead of its resolved result.",
                    confidence=0.74,
                    evidenceRefs=[_line_code_ref(task, index), *_matching_rule_ref(task, signals, preferred_type="async-risk")],
                )
            )

    for index in range(len(task.addedLines) - 1):
        line_no = _line_for_match(task, index)
        next_line_no = _line_for_match(task, index + 1)
        stripped = task.addedLines[index].strip()
        next_stripped = task.addedLines[index + 1].strip()
        if not next_stripped or next_stripped.startswith("#"):
            continue
        if PY_TERMINAL_STATEMENT.search(stripped):
            candidates.append(
                CandidateFinding(
                    filePath=task.filePath,
                    lineStart=next_line_no,
                    lineEnd=next_line_no,
                    severity="medium",
                    category="bugs",
                    title="Remove unreachable code after terminal statement",
                    body="This block adds executable code after a return or raise statement, so the later line will never run.",
                    confidence=0.9,
                    evidenceRefs=[_line_code_ref(task, index + 1), *_matching_rule_ref(task, signals, preferred_type="unreachable-after-terminal")],
                )
            )

    deduped: list[CandidateFinding] = []
    seen: set[tuple[int, str]] = set()
    for candidate in candidates:
        key = (candidate.lineStart, candidate.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped[:3]
