#!/usr/bin/env bash
# smoke_test.sh – editable-install local plugins and run a quick SDK sanity check.
#
# Usage:
#   ./smoke_test.sh
#   GLMOCR_REMOTE_OCR_API_URL=http://host:8001/v1/chat/completions ./smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLM_OCR_DIR="$(realpath "$SCRIPT_DIR/../docling-glm-ocr")"
PP_DOC_LAYOUT_DIR="$(realpath "$SCRIPT_DIR/../docling-pp-doc-layout")"
PP_OCR_DIR="$(realpath "$SCRIPT_DIR/../docling-pp-ocrv6")"

# ── 1) Editable installs ──────────────────────────────────────────────────────
echo "==> Installing plugins in editable mode ..."
echo "    docling-glm-ocr      : $GLM_OCR_DIR"
echo "    docling-pp-doc-layout: $PP_DOC_LAYOUT_DIR"
echo "    docling-pp-ocrv6     : $PP_OCR_DIR"
uv pip install docling
uv pip install --no-cache-dir "huggingface-hub>=1.3.0"
uv pip install --no-cache-dir "rapidocr>=3.0" "pyyaml>=6.0" "onnxruntime>=1.17"
uv pip install --no-deps -e "$GLM_OCR_DIR" -e "$PP_DOC_LAYOUT_DIR" -e "$PP_OCR_DIR"
uv pip install --no-cache-dir "transformers>=5.1.0"
echo ""

# ── 2) Resolve config ─────────────────────────────────────────────────────────
export GLMOCR_REMOTE_OCR_API_URL="${GLMOCR_REMOTE_OCR_API_URL:-http://localhost:8001/v1/chat/completions}"
export SMOKE_TEST_IMAGE="$SCRIPT_DIR/data/ocr.png"

echo "==> Running SDK smoke tests ..."
echo "    vLLM URL  : $GLMOCR_REMOTE_OCR_API_URL"
echo "    Test image: $SMOKE_TEST_IMAGE"
echo ""

# ── 3) Python smoke test ──────────────────────────────────────────────────────
uv run python - <<'PYTHON'
import os
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling_glm_ocr import GlmOcrRemoteOptions
from docling_pp_doc_layout.options import PPDocLayoutV3Options
from docling_pp_ocrv6.options import PPOCRv6Options

VLLM_URL = os.environ["GLMOCR_REMOTE_OCR_API_URL"]
TEST_IMAGE = Path(os.environ["SMOKE_TEST_IMAGE"])


def make_converter(*, ocr: str | None = None, layout: bool = False) -> DocumentConverter:
    do_ocr = ocr is not None
    opts = PdfPipelineOptions(
        allow_external_plugins=True,
        do_ocr=do_ocr,
        force_full_page_ocr=do_ocr,
    )
    if ocr == "glm":
        opts.ocr_options = GlmOcrRemoteOptions(api_url=VLLM_URL)
    elif ocr == "pp":
        opts.ocr_options = PPOCRv6Options()
    if layout:
        opts.layout_options = PPDocLayoutV3Options()
    return DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
    )


failures: list[str] = []

# ── Test 1: GLM-OCR only ──────────────────────────────────────────────────────
print("[1/5] GLM-OCR only ...", flush=True)
try:
    result = make_converter(ocr="glm").convert(str(TEST_IMAGE))
    md = result.document.export_to_markdown()
    assert len(md.strip()) > 0, "Expected non-empty markdown from OCR"
    print(f"      OK  ({len(md.strip())} chars)")
except Exception as exc:
    print(f"      FAIL: {exc}")
    failures.append(f"GLM-OCR: {exc}")

# ── Test 2: PP-DocLayout only (default OCR) ───────────────────────────────────
print("[2/5] PP-DocLayout only ...", flush=True)
try:
    result = make_converter(layout=True).convert(str(TEST_IMAGE))
    md = result.document.export_to_markdown()
    print(f"      OK  ({len(md.strip())} chars)")
except Exception as exc:
    print(f"      FAIL: {exc}")
    failures.append(f"PP-DocLayout: {exc}")

# ── Test 3: Both combined (GLM-OCR + PP-DocLayout) ───────────────────────────
print("[3/5] GLM-OCR + PP-DocLayout ...", flush=True)
try:
    result = make_converter(ocr="glm", layout=True).convert(str(TEST_IMAGE))
    md = result.document.export_to_markdown()
    assert len(md.strip()) > 0, "Expected non-empty markdown from combined pipeline"
    print(f"      OK  ({len(md.strip())} chars)")
except Exception as exc:
    print(f"      FAIL: {exc}")
    failures.append(f"Combined GLM + Layout: {exc}")

# ── Test 4: PP-OCR only (local) ───────────────────────────────────────────────
print("[4/5] PP-OCR only ...", flush=True)
try:
    result = make_converter(ocr="pp").convert(str(TEST_IMAGE))
    md = result.document.export_to_markdown()
    assert len(md.strip()) > 0, "Expected non-empty markdown from local PP-OCR"
    print(f"      OK  ({len(md.strip())} chars)")
except Exception as exc:
    print(f"      FAIL: {exc}")
    failures.append(f"PP-OCR: {exc}")

# ── Test 5: Both combined (PP-OCR + PP-DocLayout) ─────────────────────────────
print("[5/5] PP-OCR + PP-DocLayout ...", flush=True)
try:
    result = make_converter(ocr="pp", layout=True).convert(str(TEST_IMAGE))
    md = result.document.export_to_markdown()
    assert len(md.strip()) > 0, "Expected non-empty markdown from combined local PP-OCR + layout pipeline"
    print(f"      OK  ({len(md.strip())} chars)")
except Exception as exc:
    print(f"      FAIL: {exc}")
    failures.append(f"Combined PP-OCR + Layout: {exc}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("")
total_tests = 5
if failures:
    print(f"FAILED ({len(failures)}/{total_tests} checks):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"All {total_tests} checks passed.")
PYTHON
