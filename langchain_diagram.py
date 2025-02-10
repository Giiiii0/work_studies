import os
import asyncio
import textwrap
import PyPDF2
import re
from dotenv import load_dotenv
from graphviz import Digraph
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
STEP_BY_STEP_PROMPT = textwrap.dedent("""
    Summarize the document into a **clear and structured step-by-step process**.
    - Ensure **each step** is **numbered** (e.g., "Step 1:", "Step 2:", "Step 3:").
    - Each step should be **short and action-based**.
    - **Do NOT summarize** the entire process into a single block.
    - Provide at **least 5 steps** (if the document has enough content).
    - If no relevant information is found, return **"No relevant steps found."**
""")

# Define Summary Model
class StepByStepSummary(BaseModel):
    steps: list[str] = Field(description="A list of step-by-step instructions extracted from the document.")

# Summarizer Class
class Summarizer:
    def __init__(self, file_type):
        self.file_type = file_type
        self.summary_parser = PydanticOutputParser(pydantic_object=StepByStepSummary)

    def get_base_llm(self, streaming: bool = False):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=streaming, api_key=api_key)

    def format_docs_with_ids(self, docs: list[Document]) -> str:
        formatted = [f"Content ID: {doc.metadata['ContentID']}, Source ID: {doc.metadata['ChunkID']}\n{doc.page_content}" for doc in docs]
        return "\n\n" + "\n\n".join(formatted)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
    async def base_summarize(self, docs: list[Document], question: str) -> StepByStepSummary:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_ROLE),
            ("human", "{chain_input}"),
        ])
        llm = self.get_base_llm()
        chain = prompt_template | llm | self.summary_parser
        chain_input = textwrap.dedent(f"""
            Question: {question}
            {STEP_BY_STEP_PROMPT}

            {self.summary_parser.get_format_instructions()}

            Documents:
            {self.format_docs_with_ids(docs)}
        """)
        response = await chain.ainvoke({"chain_input": chain_input})
        return response

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Function to generate a step-by-step flowchart from summarized steps
def create_flowchart(steps):
    dot = Digraph(format="png")

    # Ensure we have valid steps
    if not steps or steps == ["No relevant steps found."]:
        steps = ["No steps found. Please check the input document."]

    # Add start node
    dot.node("Start", "Start", shape="oval", style="filled", fillcolor="lightgreen")

    # Add process steps
    for i, step in enumerate(steps):
        step_label = textwrap.fill(step, width=40)  # Wrap text for readability
        dot.node(f"Step {i+1}", step_label, shape="rectangle", style="filled", fillcolor="lightblue")

    # Add end node
    dot.node("End", "End", shape="oval", style="filled", fillcolor="lightgreen")

    # Connect nodes sequentially
    dot.edge("Start", "Step 1")  # Start to first step
    for i in range(len(steps) - 1):
        dot.edge(f"Step {i+1}", f"Step {i+2}")  # Connect each step
    dot.edge(f"Step {len(steps)}", "End")  # Last step to end

    # Render and return file path
    flowchart_path = "step_by_step_summary_flowchart"
    dot.render(flowchart_path)
    return f"{flowchart_path}.png"

# Path to the PDF file
pdf_path = "stepbystepguide.pdf"

# Initialize summarizer
summarizer = Summarizer(file_type="PDF")

# Extract text from the PDF
document_text = extract_text_from_pdf(pdf_path)

# Convert extracted text into Document objects
documents = [Document(page_content=page, metadata={"ChunkID": idx, "ContentID": None}) for idx, page in enumerate(document_text.split("\n\n"))]

# Run summarization and generate flowchart
async def run_summarization():
    summary = await summarizer.base_summarize(documents, question="Summarize the step-by-step process")
    
    # Ensure LLM output is correctly formatted as steps
    steps = summary.steps

    # If LLM outputs a block instead of steps, manually extract steps using regex
    if len(steps) == 1 and not steps[0].startswith("Step"):
        print("⚠️ WARNING: LLM output is not properly formatted as steps. Attempting to extract manually.")
        extracted_steps = re.findall(r"Step \d+:.*?(?=Step \d+:|$)", steps[0], re.DOTALL)
        steps = [s.strip() for s in extracted_steps] if extracted_steps else steps

    print("✅ Extracted Steps:", steps)  # Debugging: Print steps

    # Generate flowchart from summarized content
    flowchart_path = create_flowchart(steps)
    print(f"📌 Flowchart saved at: {flowchart_path}")

# Execute the summarization
asyncio.run(run_summarization())
