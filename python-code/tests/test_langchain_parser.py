import sys
import os
import pytest
from unittest.mock import AsyncMock
from langchain_core.documents.base import Document
from langchain_core.output_parsers import PydanticOutputParser

# Add the parent directory (python-code) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_parser import extract_text_from_pdf, Summarizer, BulletPointSummary

# Mock the environment variable for API Key
@pytest.fixture(autouse=True)
def mock_env_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

# Test: PDF Text Extraction
def test_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    # Create a sample PDF file
    from reportlab.pdfgen import canvas
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(100, 750, "Test content in PDF")
    pdf.save()

    extracted_text = extract_text_from_pdf(str(pdf_path))
    assert "Test content in PDF" in extracted_text, "PDF text extraction failed"

# Test: Summarizer Initialization
def test_summarizer_initialization():
    summarizer = Summarizer(file_type="PDF")
    assert summarizer.file_type == "PDF", "Summarizer file type not set correctly"
    assert isinstance(summarizer.summary_parser, PydanticOutputParser), "Summarizer parser not initialized correctly"

# Test: Format Documents
def test_format_docs_with_ids():
    summarizer = Summarizer(file_type="PDF")
    docs = [
        Document(page_content="Sample text 1", metadata={"ContentID": "001", "ChunkID": "A"}),
        Document(page_content="Sample text 2", metadata={"ContentID": "002", "ChunkID": "B"}),
    ]
    formatted = summarizer.format_docs_with_ids(docs)
    assert "Content ID: 001, Source ID: A" in formatted, "Document formatting failed"

# Test: Summarization Output
@pytest.mark.asyncio
async def test_base_summarize():
    summarizer = Summarizer(file_type="PDF")
    
    # Mock LLM response
    mock_response = BulletPointSummary(bullets=["Step 1: Test bullet", "Step 2: Another bullet"])
    
    summarizer.base_summarize = AsyncMock(return_value=mock_response)

    docs = [Document(page_content="Test content", metadata={"ChunkID": "1", "ContentID": "Doc1"})]
    result = await summarizer.base_summarize(docs, question="Summarize the document")

    assert isinstance(result, BulletPointSummary), "Summarization output format incorrect"
    assert len(result.bullets) > 0, "No bullets returned"
    assert result.bullets[0].startswith("Step"), "Output does not follow step formatting"

# Run the tests if executed directly
if __name__ == "__main__":
    pytest.main()
