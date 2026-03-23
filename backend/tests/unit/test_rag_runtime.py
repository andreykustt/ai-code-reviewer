from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_SRC = REPO_ROOT / "rag-ml" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from rag_ml.citation_resolver import CitationResolver
from rag_ml.context_builder import build_context_pack
from rag_ml.bug_rules import rule_based_bug_candidates
from rag_ml.evidence_models import code_ref, doc_ref
from rag_ml.file_classifier import classify_file
from rag_ml.hotspot_planner import plan_hotspot_tasks
from rag_ml.kb_chunker import chunk_documents
from rag_ml.kb_normalizer import normalize_descriptor
from rag_ml.language_mapper import to_slug
from rag_ml.pr_overview import build_pr_overview
from rag_ml.query_builder import build_query
from rag_ml.rule_fallbacks import style_fallback_candidates
from rag_ml.schemas import CandidateFinding, DocumentDescriptor, HunkTask, KnowledgeChunk, NormalizedDocument, PROverview, RagRequest, RagFile, RetrievalHit, SectionSpan
from rag_ml.schemas import StaticSignal
from rag_ml.generator import SuggestionGenerator
from rag_ml.ollama_client import OllamaStructuredOutputError
from rag_ml.static_signals import collect_static_signals
from rag_ml.validator import SuggestionValidator
from rag_ml.verifier import FindingVerifier


class _OverviewClient:
    async def chat_structured(self, messages, schema, **kwargs):
        return {
            "prIntent": "Refactor auth flow",
            "riskLevel": "medium",
            "recommendedScopes": ["style", "bugs"],
            "hotspots": [{"filePath": "lib/example.dart", "reasons": ["async flow changed"], "risk": 0.8}],
            "notes": ["Touches async flow"],
        }


class _GeneratorClient:
    repair_model = "gemma3:12b"

    async def chat_structured(self, messages, schema, **kwargs):
        content = messages[0].content
        if "syntax repair assistant" in content:
            return {
                "findings": [
                    {
                        "filePath": "src/example.py",
                        "lineStart": 10,
                        "lineEnd": 10,
                        "severity": "warning",
                        "category": "bug",
                        "shortLabel": "broad exception hides failures",
                        "confidence": 0.8,
                        "evidenceRefs": ["rule:src/example.py:10:broad-except"],
                    }
                ]
            }
        raise AssertionError("unexpected structured call in test")

    async def chat_text(self, messages, **kwargs):
        return "FINDING|bug|warning|10|10|broad exception hides failures|rule:src/example.py:10:broad-except"


class _RepairingGeneratorClient:
    repair_model = "gemma3:12b"

    def __init__(self) -> None:
        self.calls = 0

    async def chat_structured(self, messages, schema, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise OllamaStructuredOutputError(
                "Ollama returned invalid JSON: missing comma",
                '{"findings": [{"category": "bug" "severity": "warning"}]}',
            )
        return {
            "findings": [
                {
                    "filePath": "src/example.py",
                    "lineStart": 10,
                    "lineEnd": 10,
                    "severity": "warning",
                    "category": "bug",
                    "shortLabel": "broad exception hides failures",
                    "confidence": 0.8,
                    "evidenceRefs": ["rule:src/example.py:10:broad-except"],
                }
            ]
        }

    async def chat_text(self, messages, **kwargs):
        return "NO_FINDINGS"


class RagRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_language_mapper_maps_supported_display_names(self) -> None:
        self.assertEqual(to_slug("Python"), "python")
        self.assertEqual(to_slug("Dart"), "dart")
        self.assertEqual(to_slug("Swift"), "swift")
        self.assertEqual(to_slug("C++"), "cpp")
        self.assertEqual(to_slug("JavaScript"), "javascript")

    def test_chunker_preserves_metadata_and_size(self) -> None:
        document = NormalizedDocument(
            documentId="dart:effective-dart:sample",
            namespace="dart",
            language="dart",
            sourceId="effective-dart",
            sourceTitle="Effective Dart",
            sourceUrl="https://dart.dev/guides/language/effective-dart",
            docPath="/tmp/effective-dart.txt",
            text=("Style section\n\n" + ("lowerCamelCase is recommended. " * 200)),
            sections=[SectionSpan(headingPath=["Effective Dart", "Style"], charStart=0, charEnd=len("Style section\n\n" + ("lowerCamelCase is recommended. " * 200)))],
        )
        chunks = chunk_documents([document])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].sourceId, "effective-dart")
        self.assertEqual(chunks[0].sourceUrl, "https://dart.dev/guides/language/effective-dart")
        self.assertLessEqual(max(len(chunk.text) for chunk in chunks), 2200)
        self.assertLess(len(chunks), 10)

    def test_citation_resolver_uses_real_chunk_text(self) -> None:
        chunk = KnowledgeChunk(
            chunkId="dart:effective-dart:000001",
            namespace="dart",
            language="dart",
            sourceId="effective-dart",
            sourceTitle="Effective Dart",
            sourceUrl="https://dart.dev/guides/language/effective-dart",
            docPath="/tmp/effective-dart.txt",
            headingPath=["Effective Dart", "Style"],
            text="Use lowerCamelCase for variables and function names to preserve consistency across Dart codebases.",
            charStart=0,
            charEnd=96,
            tokenEstimate=24,
        )
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_var = 1;",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+final My_var = 1;",
            addedLines=["final My_var = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        context = build_context_pack(task, [], [RetrievalHit(
            chunkId=chunk.chunkId,
            namespace=chunk.namespace,
            sourceId=chunk.sourceId,
            title=chunk.sourceTitle,
            url=chunk.sourceUrl,
            headingPath=chunk.headingPath,
            text=chunk.text,
            finalScore=0.9,
            sparseRank=1,
            denseRank=1,
        )])
        evidence, citations = CitationResolver({chunk.chunkId: chunk}).resolve([doc_ref(chunk.chunkId)], context)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].sourceId, "effective-dart")
        self.assertIn("lowerCamelCase", citations[0].snippet)
        self.assertEqual(evidence[0].type, "doc")

    def test_citation_resolver_preserves_multiline_doc_snippets_for_evidence_blocks(self) -> None:
        chunk = KnowledgeChunk(
            chunkId="python:asyncio:000777",
            namespace="python",
            language="python",
            sourceId="python-docs",
            sourceTitle="Python Docs",
            sourceUrl="https://docs.python.org/3/library/asyncio.html",
            docPath="/tmp/asyncio.txt",
            headingPath=["Python Docs", "asyncio", "Examples"],
            text="await client.fetch()\nresult = normalize(payload)\n\nUse await for IO-bound calls inside coroutines.",
            charStart=0,
            charEnd=92,
            tokenEstimate=20,
        )
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_var = 1;",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+final My_var = 1;",
            addedLines=["final My_var = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        context = build_context_pack(task, [], [RetrievalHit(
            chunkId=chunk.chunkId,
            namespace=chunk.namespace,
            sourceId=chunk.sourceId,
            title=chunk.sourceTitle,
            url=chunk.sourceUrl,
            headingPath=chunk.headingPath,
            text=chunk.text,
            finalScore=0.9,
            sparseRank=1,
            denseRank=1,
        )])

        evidence, _ = CitationResolver({chunk.chunkId: chunk}).resolve([doc_ref(chunk.chunkId)], context)

        self.assertEqual(evidence[0].snippet.splitlines()[0], "await client.fetch()")
        self.assertIn("result = normalize(payload)", evidence[0].snippet)
        self.assertEqual(evidence[0].metadata.get("docPath"), "/tmp/asyncio.txt")

    def test_build_context_pack_creates_focused_code_window_with_highlight(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch=(
                "@@ -20,11 +20,11 @@\n"
                " line_20\n"
                " line_21\n"
                " line_22\n"
                " line_23\n"
                " line_24\n"
                "+problem_line()\n"
                " line_26\n"
                " line_27\n"
                " line_28\n"
                " line_29\n"
                " line_30"
            ),
            hunkIndex=0,
            hunkHeader="@@ -20,11 +20,11 @@",
            hunkPatch=(
                "@@ -20,11 +20,11 @@\n"
                " line_20\n"
                " line_21\n"
                " line_22\n"
                " line_23\n"
                " line_24\n"
                "+problem_line()\n"
                " line_26\n"
                " line_27\n"
                " line_28\n"
                " line_29\n"
                " line_30"
            ),
            addedLines=["problem_line()"],
            changedNewLines=[25],
            firstChangedLine=25,
            priority=1.0,
        )

        context = build_context_pack(task, [], [])
        primary_candidate = context.codeEvidenceCandidates[0]
        window = primary_candidate.metadata.get("contextWindow")

        self.assertEqual(primary_candidate.lineStart, 25)
        self.assertIsInstance(window, list)
        self.assertEqual(len(window), 11)
        self.assertEqual(window[5]["lineNumber"], 25)
        self.assertTrue(window[5]["highlight"])
        self.assertEqual(window[5]["text"], "problem_line()")

    def test_validator_rejects_missing_evidence(self) -> None:
        candidate = CandidateFinding(
            filePath="lib/example.dart",
            lineStart=10,
            lineEnd=10,
            severity="low",
            category="style",
            title="Use lowerCamelCase",
            body="Rename the symbol to lowerCamelCase for consistency.",
            confidence=0.8,
            evidenceRefs=[],
        )
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_var = 1;",
            hunkIndex=0,
            hunkHeader="",
            hunkPatch="@@ -1 +1 @@\n+final My_var = 1;",
            addedLines=["final My_var = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        result = SuggestionValidator().validate(candidate, task, {"style"})
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "missing_evidence")

    def test_txt_normalizer_prefers_rst_headings_over_plain_paragraph_lines(self) -> None:
        sample = "\n".join(
            [
                "9. Classes",
                "**********",
                "",
                "Classes provide a means of bundling data and functionality together.",
                "Creating a new class creates a new type of object.",
                "",
                "9.1. A Word About Names and Objects",
                "===================================",
                "",
                "Objects can contain arbitrary amounts and kinds of data.",
                "",
            ]
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classes.txt"
            path.write_text(sample, encoding="utf-8")
            document = normalize_descriptor(
                DocumentDescriptor(
                    namespace="python",
                    language="python",
                    displayName="Python",
                    sourceId="python-docs",
                    sourceTitle="Python Tutorial",
                    sourceUrl="https://docs.python.org/3/tutorial/",
                    docPath=str(path),
                    isReadme=False,
                )
            )

        self.assertEqual(len(document.sections), 2)
        self.assertEqual(document.sections[0].headingPath, ["Python Tutorial", "9. Classes"])
        self.assertEqual(document.sections[1].headingPath, ["Python Tutorial", "9.1. A Word About Names and Objects"])

    def test_query_builder_adds_style_hints_for_bad_identifier_names(self) -> None:
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_value = 1;",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+final My_value = 1;",
            addedLines=["final My_value = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )

        query = build_query(task, "style")
        self.assertIn("lowerCamelCase", query)
        self.assertIn("UpperCamelCase", query)

    def test_validator_rejects_generic_documentation_summary(self) -> None:
        candidate = CandidateFinding(
            filePath="lib/example.dart",
            lineStart=1,
            lineEnd=1,
            severity="info",
            category="style",
            title="Dart language tour overview",
            body="This section provides an overview of the Dart programming language and its features.",
            confidence=1.0,
            evidenceRefs=[doc_ref("dart:effective-dart:000001")],
        )
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_value = 1;",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+final My_value = 1;",
            addedLines=["final My_value = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )

        result = SuggestionValidator().validate(candidate, task, {"style"})
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "generic_feedback")

    def test_file_classifier_marks_localization_as_resource(self) -> None:
        file = RagFile(
            path="lib/presentation/localization/control_strings_en.dart",
            language="Dart",
            patch="@@ -1 +1 @@\n+final label = 'ok';",
        )
        self.assertEqual(classify_file(file), "resource")

    def test_bug_rules_detect_mutable_default_argument(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -1 +1 @@\n+def build_items(items=[]):",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+def build_items(items=[]):",
            addedLines=["def build_items(items=[]):"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        candidates = rule_based_bug_candidates(task, [])
        self.assertTrue(any(candidate.title == "Avoid mutable default arguments" for candidate in candidates))

    def test_bug_rules_detect_unreachable_after_return(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -1 +1 @@\n+return item\n+cache.invalidate(key)",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+return item\n+cache.invalidate(key)",
            addedLines=["return item", "cache.invalidate(key)"],
            changedNewLines=[10, 11],
            firstChangedLine=10,
            priority=1.0,
        )
        signals = [
            StaticSignal(
                signalId="unreachable-after-terminal",
                filePath="src/example.py",
                type="unreachable-after-terminal",
                severity="medium",
                message="Code after return may be unreachable.",
                lineStart=10,
                lineEnd=11,
            )
        ]
        candidates = rule_based_bug_candidates(task, signals)
        self.assertTrue(any(candidate.title == "Remove unreachable code after terminal statement" for candidate in candidates))

    def test_bug_rules_point_unreachable_code_evidence_at_offending_line(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -8,0 +10,3 @@\n+value = load()\n+raise ValueError('boom')\n+persist(value)",
            hunkIndex=0,
            hunkHeader="@@ -8,0 +10,3 @@",
            hunkPatch="@@ -8,0 +10,3 @@\n+value = load()\n+raise ValueError('boom')\n+persist(value)",
            addedLines=["value = load()", "raise ValueError('boom')", "persist(value)"],
            changedNewLines=[10, 11, 12],
            firstChangedLine=10,
            priority=1.0,
        )

        candidates = rule_based_bug_candidates(task, [])
        unreachable = next(candidate for candidate in candidates if candidate.title == "Remove unreachable code after terminal statement")
        self.assertEqual(unreachable.lineStart, 12)
        self.assertEqual(unreachable.evidenceRefs[0], code_ref(task.taskId, 3))

    def test_context_pack_builds_line_focused_code_candidates(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -8,0 +10,3 @@\n+value = load()\n+raise ValueError('boom')\n+persist(value)",
            hunkIndex=0,
            hunkHeader="@@ -8,0 +10,3 @@",
            hunkPatch="@@ -8,0 +10,3 @@\n+value = load()\n+raise ValueError('boom')\n+persist(value)",
            addedLines=["value = load()", "raise ValueError('boom')", "persist(value)"],
            changedNewLines=[10, 11, 12],
            firstChangedLine=10,
            priority=1.0,
        )

        context = build_context_pack(task, [], [])
        code_candidate = next(candidate for candidate in context.codeEvidenceCandidates if candidate.refId == code_ref(task.taskId, 3))
        self.assertEqual(code_candidate.lineStart, 12)
        self.assertIn("raise ValueError('boom')", code_candidate.snippet)
        self.assertIn("persist(value)", code_candidate.snippet)

    def test_verifier_rejects_valid_private_dart_type_name(self) -> None:
        candidate = CandidateFinding(
            filePath="lib/example.dart",
            lineStart=1,
            lineEnd=1,
            severity="low",
            category="style",
            title="Use UpperCamelCase for type names",
            body="`_LogoWidget` should use UpperCamelCase.",
            confidence=0.82,
            evidenceRefs=["code:lib/example.dart:0:0"],
        )
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+class _LogoWidget extends StatelessWidget {",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+class _LogoWidget extends StatelessWidget {",
            addedLines=["class _LogoWidget extends StatelessWidget {"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        result = FindingVerifier().verify(candidate, task, {"style"}, build_context_pack(task, [], []))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "style_already_valid")

    def test_style_fallback_candidates_generate_grounded_dart_naming_comment(self) -> None:
        task = HunkTask(
            taskId="lib/example.dart:0",
            filePath="lib/example.dart",
            language="Dart",
            languageSlug="dart",
            patch="@@ -1 +1 @@\n+final My_value = 1;",
            hunkIndex=0,
            hunkHeader="@@ -1 +1 @@",
            hunkPatch="@@ -1 +1 @@\n+final My_value = 1;",
            addedLines=["final My_value = 1;"],
            changedNewLines=[1],
            firstChangedLine=1,
            priority=1.0,
        )
        hits = [
            RetrievalHit(
                chunkId="dart:effective-dart:000005",
                namespace="dart",
                sourceId="effective-dart",
                title="Effective Dart",
                url="https://dart.dev/guides/language/effective-dart",
                headingPath=["Effective Dart"],
                text="DO name other identifiers using lowerCamelCase.",
                finalScore=0.9,
                sparseRank=1,
                denseRank=1,
            )
        ]

        candidates = style_fallback_candidates(task, hits)
        self.assertEqual(len(candidates), 1)
        self.assertIn("lowerCamelCase", candidates[0].body)
        self.assertEqual(candidates[0].evidenceRefs, [doc_ref("dart:effective-dart:000005")])

    def test_static_signals_and_planner_create_tasks(self) -> None:
        request = RagRequest(
            jobId="job_1",
            snapshotId="snap_1",
            prId="pr_1",
            title="Auth refactor",
            scope=["bugs", "style", "performance"],
            files=[
                RagFile(
                    path="lib/auth/service.dart",
                    language="Dart",
                    patch="@@ -1 +1 @@\n+await client.refreshToken();",
                    surroundingCode=[{"lineNumber": 1, "text": "await client.refreshToken();"}],
                )
            ],
            limits={"maxComments": 10, "maxPerFile": 3},
        )
        overview = PROverview(
            prIntent="Auth refactor",
            riskLevel="medium",
            recommendedScopes=["bugs", "style"],
            hotspots=[{"filePath": "lib/auth/service.dart", "reasons": ["auth-change"], "risk": 0.8}],
            notes=[],
        )
        static_checks = collect_static_signals(request.files)
        tasks = plan_hotspot_tasks(request, overview, static_checks, max_hunks_per_file=2, max_hotspot_tasks=8)
        self.assertGreaterEqual(len(tasks), 1)
        self.assertIn("bugs", tasks[0].categories)

    async def test_pr_overview_returns_structured_result(self) -> None:
        request = RagRequest(
            jobId="job_1",
            snapshotId="snap_1",
            prId="pr_1",
            title="Auth refactor",
            description="Touches token refresh flow",
            scope=["bugs", "style"],
            files=[RagFile(path="lib/auth/service.dart", language="Dart", patch="@@ -1 +1 @@\n+await client.refreshToken();")],
            limits={"maxComments": 10, "maxPerFile": 3},
        )
        overview = await build_pr_overview(_OverviewClient(), request)
        self.assertEqual(overview.riskLevel, "medium")
        self.assertEqual(overview.hotspots[0].filePath, "lib/example.dart")

    async def test_generator_line_fallback_normalizes_aliases(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -10 +10 @@\n+except Exception:",
            hunkIndex=0,
            hunkHeader="@@ -10 +10 @@",
            hunkPatch="@@ -10 +10 @@\n+except Exception:",
            addedLines=["except Exception:"],
            changedNewLines=[10],
            firstChangedLine=10,
            priority=1.0,
        )
        context = build_context_pack(
            task,
            [
                StaticSignal(
                    signalId="broad-except",
                    filePath="src/example.py",
                    type="broad-except",
                    severity="medium",
                    message="Avoid broad exception handling where possible.",
                    lineStart=10,
                    lineEnd=10,
                )
            ],
            [
                RetrievalHit(
                    chunkId="python:docs:000001",
                    namespace="python",
                    sourceId="python-docs",
                    title="Python docs",
                    url="https://docs.python.org/3/tutorial/errors.html",
                    headingPath=["Errors and Exceptions"],
                    text="Avoid overly broad exception handling where possible.",
                    finalScore=0.7,
                    sparseRank=1,
                    denseRank=1,
                )
            ],
        )
        context.ruleEvidenceCandidates = [
            context.ruleEvidenceCandidates[0].model_copy(update={"refId": "rule:src/example.py:10:broad-except"})
        ]

        generator = SuggestionGenerator(_GeneratorClient())
        envelope = await generator._detect_with_line_format(task, ["bugs"], context, max_findings=2)
        self.assertIsNotNone(envelope)
        self.assertEqual(len(envelope.findings), 1)
        self.assertEqual(envelope.findings[0].category, "bugs")
        self.assertEqual(envelope.findings[0].severity, "medium")

    async def test_generator_repair_stage_recovers_invalid_json(self) -> None:
        task = HunkTask(
            taskId="src/example.py:0",
            filePath="src/example.py",
            language="Python",
            languageSlug="python",
            patch="@@ -10 +10 @@\n+except Exception:",
            hunkIndex=0,
            hunkHeader="@@ -10 +10 @@",
            hunkPatch="@@ -10 +10 @@\n+except Exception:",
            addedLines=["except Exception:"],
            changedNewLines=[10],
            firstChangedLine=10,
            priority=1.0,
        )
        context = build_context_pack(
            task,
            [
                StaticSignal(
                    signalId="broad-except",
                    filePath="src/example.py",
                    type="broad-except",
                    severity="medium",
                    message="Avoid broad exception handling where possible.",
                    lineStart=10,
                    lineEnd=10,
                )
            ],
            [],
        )
        context.ruleEvidenceCandidates = [
            context.ruleEvidenceCandidates[0].model_copy(update={"refId": "rule:src/example.py:10:broad-except"})
        ]

        generator = SuggestionGenerator(_RepairingGeneratorClient())
        envelope = await generator.detect(task, ["bugs"], context, max_findings=2)
        self.assertEqual(len(envelope.findings), 1)
        self.assertEqual(envelope.findings[0].category, "bugs")
        self.assertEqual(envelope.findings[0].severity, "medium")


if __name__ == "__main__":
    unittest.main()
