"""Example: convert a PDF using both docling plugins (GLM-OCR + PP-DocLayout-V3).

Prerequisites:
    pip install docling docling-glm-ocr docling-pp-doc-layout

    A running vLLM server hosting zai-org/GLM-OCR is required for the OCR
    plugin.  See the project README for docker run instructions.
"""

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_glm_ocr import GlmOcrRemoteOptions
from docling_pp_doc_layout.options import PPDocLayoutV3Options

pipeline_options = PdfPipelineOptions(
    allow_external_plugins=True,
    ocr_options=GlmOcrRemoteOptions(
        api_url="http://localhost:8001/v1/chat/completions",
        model_name="zai-org/GLM-OCR",
    ),
    layout_options=PPDocLayoutV3Options(),
)

converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
result = converter.convert("https://arxiv.org/pdf/2501.17887")
print(result.document.export_to_markdown())
