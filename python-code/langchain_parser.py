import os
import re
import asyncio
import textwrap
import pypdf
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Constants for summarization
AGENT_ROLE = "You're an expert compliance officer, risk manager, auditor, and legal advisor"
DEFAULT_BULLET_SUMMARY = textwrap.dedent("""
    Summarize the key points of the document in **3 to 4 bullet points**.
    - Each bullet point should be **clear and concise**.
    - **Do NOT return a single-block summary**.
    - Format the response as **Step 1:, Step 2:, Step 3: etc.**.
    - If no relevant information is found, return **'No relevant information found.'**
""")

# Define Structured Bullet-Point Output Model
class BulletPointSummary(BaseModel):
    bullets: list[str] = Field(description="A structured bullet-point summary of the document.")

# Summarizer Class
class Summarizer:
    def __init__(self, file_type):
        self.file_type = file_type
        self.summary_parser = PydanticOutputParser(pydantic_object=BulletPointSummary)

    def get_base_llm(self, streaming: bool = False):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=streaming, api_key=api_key)

    def format_docs_with_ids(self, docs: list[Document]) -> str:
        formatted = [f"Content ID: {doc.metadata['ContentID']}, Source ID: {doc.metadata['ChunkID']}\n{doc.page_content}" for doc in docs]
        return "\n\n" + "\n\n".join(formatted)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
    async def base_summarize(self, docs: list[Document], question: str) -> BulletPointSummary:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_ROLE),
            ("human", "{chain_input}"),
        ])
        llm = self.get_base_llm()
        chain = prompt_template | llm | self.summary_parser
        chain_input = textwrap.dedent(f"""
            Question: {question}
            {DEFAULT_BULLET_SUMMARY}

            {self.summary_parser.get_format_instructions()}

            Documents:
            {self.format_docs_with_ids(docs)}
        """)
        response = await chain.ainvoke({"chain_input": chain_input})
        return response

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Path to the PDF file
pdf_path = os.path.join(os.path.dirname(__file__), "policy_template.pdf")

# Initialize summarizer
summarizer = Summarizer(file_type="PDF")

# Extract text from the PDF
document_text = extract_text_from_pdf(pdf_path)

# Convert extracted text into Document objects
documents = [Document(page_content=page, metadata={"ChunkID": idx, "ContentID": None}) for idx, page in enumerate(document_text.split("\n\n"))]

# Run summarization and verify output format
async def run_summarization():
    summary = await summarizer.base_summarize(documents, question="Summarize the document")

    # If the response is still in block format, attempt to split it into bullet points
    if len(summary.bullets) == 1 and not summary.bullets[0].startswith("Step"):
        print("⚠️ WARNING: LLM output is not formatted correctly. Attempting to extract manually.")
        extracted_bullets = re.findall(r"Step \d+:.*?(?=Step \d+:|$)", summary.bullets[0], re.DOTALL)
        summary.bullets = [s.strip() for s in extracted_bullets] if extracted_bullets else summary.bullets

    print("\n✅ **Bullet Point Summary:**\n")
    for idx, bullet in enumerate(summary.bullets, start=1):
        print(f"{idx}. {bullet}")

# Execute the summarization
asyncio.run(run_summarization())
