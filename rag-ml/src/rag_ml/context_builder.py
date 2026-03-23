from __future__ import annotations

import re
from typing import Any

from .evidence_models import code_ref, doc_ref, rule_ref
from .schemas import ContextEvidenceCandidate, ContextPack, HunkTask, RetrievalHit, StaticSignal


def _truncate(value: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_patch_window(hunk_patch: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    current_new = 1
    for raw_line in hunk_patch.splitlines():
        if raw_line.startswith("@@"):
            match = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", raw_line)
            if match:
                current_new = int(match.group(3))
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            lines.append({"lineNumber": current_new, "text": raw_line[1:], "kind": "add"})
            current_new += 1
            continue

        if raw_line.startswith(" "):
            lines.append({"lineNumber": current_new, "text": raw_line[1:], "kind": "ctx"})
            current_new += 1
            continue

        if raw_line.startswith("-") and not raw_line.startswith("---"):
            continue

    return lines


def _build_context_window(
    hunk_patch: str,
    *,
    line_start: int,
    line_end: int,
    before: int = 5,
    after: int = 5,
) -> list[dict[str, Any]]:
    parsed = _parse_patch_window(hunk_patch)
    if not parsed:
        return []

    matched_indices = [
        index
        for index, line in enumerate(parsed)
        if line_start <= int(line["lineNumber"]) <= line_end
    ]
    if matched_indices:
        start = max(0, matched_indices[0] - before)
        end = min(len(parsed), matched_indices[-1] + after + 1)
        window = parsed[start:end]
    else:
        window = parsed[: before + after + 1]

    return [
        {
            "lineNumber": int(line["lineNumber"]),
            "text": str(line["text"]),
            "kind": str(line["kind"]),
            "highlight": line_start <= int(line["lineNumber"]) <= line_end,
        }
        for line in window
    ]


def _window_snippet(window: list[dict[str, Any]]) -> str:
    return "\n".join(str(line["text"]) for line in window if str(line["text"]).strip())


def _fallback_window(line_start: int, line_end: int, snippet: str) -> list[dict[str, Any]]:
    rows = [row for row in snippet.splitlines() if row.strip()]
    if not rows:
        rows = [snippet]
    window: list[dict[str, Any]] = []
    current_line = line_start
    for index, row in enumerate(rows):
        line_no = current_line + index
        window.append(
            {
                "lineNumber": line_no,
                "text": row,
                "kind": "ctx",
                "highlight": line_start <= line_no <= line_end,
            }
        )
    return window


def build_context_pack(task: HunkTask, signals: list[StaticSignal], doc_hits: list[RetrievalHit]) -> ContextPack:
    code_candidates: list[ContextEvidenceCandidate] = []
    next_index = 0

    primary_line = task.firstChangedLine
    primary_window = _build_context_window(task.hunkPatch, line_start=primary_line, line_end=primary_line)
    code_candidates.append(
        ContextEvidenceCandidate(
            refId=code_ref(task.taskId, next_index),
            type="code",
            title=f"Измененный hunk {task.filePath}:{primary_line}",
            snippet=_truncate(_window_snippet(primary_window) or task.hunkPatch, 420),
            filePath=task.filePath,
            lineStart=primary_line,
            lineEnd=primary_line,
            metadata={
                "taskId": task.taskId,
                "hunkHeader": task.hunkHeader,
                "contextWindow": primary_window,
            },
        )
    )
    next_index += 1

    for added_index, line_no in enumerate(task.changedNewLines):
        line_window = _build_context_window(task.hunkPatch, line_start=line_no, line_end=line_no, before=2, after=2)
        code_candidates.append(
            ContextEvidenceCandidate(
                refId=code_ref(task.taskId, next_index),
                type="code",
                title=f"Измененная строка {task.filePath}:{line_no}",
                snippet=_truncate(_window_snippet(line_window) or task.addedLines[added_index], 240),
                filePath=task.filePath,
                lineStart=line_no,
                lineEnd=line_no,
                metadata={
                    "taskId": task.taskId,
                    "addedLineIndex": added_index,
                    "contextWindow": line_window,
                },
            )
        )
        next_index += 1

    for block in task.changedBlocks[:4]:
        context_window = _build_context_window(task.hunkPatch, line_start=block.lineStart, line_end=block.lineEnd)
        code_candidates.append(
            ContextEvidenceCandidate(
                refId=code_ref(task.taskId, next_index),
                type="code",
                title=f"Измененный блок {block.symbol or task.filePath}",
                snippet=_truncate(_window_snippet(context_window) or block.afterSnippet or block.snippet, 420),
                filePath=task.filePath,
                lineStart=block.lineStart,
                lineEnd=block.lineEnd,
                metadata={
                    "taskId": task.taskId,
                    "blockId": block.blockId,
                    "symbol": block.symbol,
                    "kind": block.kind,
                    "beforeSnippet": block.beforeSnippet,
                    "contextWindow": context_window,
                },
            )
        )
        next_index += 1
    for call_site in task.relatedCallSites[:6]:
        context_window = _fallback_window(call_site.lineStart, call_site.lineEnd, call_site.snippet)
        code_candidates.append(
            ContextEvidenceCandidate(
                refId=code_ref(task.taskId, next_index),
                type="code",
                title=f"Связанный вызов {call_site.symbol}",
                snippet=_truncate(call_site.snippet, 320),
                filePath=call_site.filePath,
                lineStart=call_site.lineStart,
                lineEnd=call_site.lineEnd,
                metadata={
                    "taskId": task.taskId,
                    "relation": call_site.relation,
                    "symbol": call_site.symbol,
                    "contextWindow": context_window,
                },
            )
        )
        next_index += 1
    if not code_candidates:
        fallback_window = _build_context_window(
            task.hunkPatch,
            line_start=task.firstChangedLine,
            line_end=max(task.changedNewLines) if task.changedNewLines else task.firstChangedLine,
        )
        code_candidates.append(
            ContextEvidenceCandidate(
                refId=code_ref(task.taskId, 0),
                type="code",
                title=f"Измененный hunk {task.filePath}",
                snippet=_truncate(_window_snippet(fallback_window) or task.hunkPatch, 360),
                filePath=task.filePath,
                lineStart=task.firstChangedLine,
                lineEnd=max(task.changedNewLines) if task.changedNewLines else task.firstChangedLine,
                metadata={"taskId": task.taskId, "contextWindow": fallback_window},
            )
        )

    rule_candidates = [
        ContextEvidenceCandidate(
            refId=rule_ref(task.taskId, index),
            type="rule",
            title=f"Статический сигнал: {signal.type}",
            snippet=_truncate(signal.message),
            filePath=signal.filePath,
            lineStart=signal.lineStart,
            lineEnd=signal.lineEnd,
            metadata={"signalId": signal.signalId, "severity": signal.severity},
        )
        for index, signal in enumerate(signals)
    ]

    doc_candidates = [
        ContextEvidenceCandidate(
            refId=doc_ref(hit.chunkId),
            type="doc",
            title=hit.title,
            snippet=_truncate(hit.text, 300),
            sourceId=hit.sourceId,
            url=hit.url,
            metadata={"chunkId": hit.chunkId, "headingPath": hit.headingPath},
        )
        for hit in doc_hits
    ]

    return ContextPack(
        taskId=task.taskId,
        codeEvidenceCandidates=code_candidates,
        ruleEvidenceCandidates=rule_candidates,
        docEvidenceCandidates=doc_candidates,
        historyEvidenceCandidates=[],
    )
