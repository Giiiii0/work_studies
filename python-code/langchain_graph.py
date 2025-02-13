import os
import asyncio
import textwrap
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict
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
AGENT_ROLE = "You're an expert data analyst. Your task is to extract structured numerical data for visualization."
DATA_EXTRACTION_PROMPT = textwrap.dedent("""
    Extract all numerical datasets from the document and structure them properly.
    Identify datasets that can be plotted (e.g., Time vs. Distance, Speed vs. Date, etc.).
    Format the response as structured lists or tables to ensure proper graph generation.
    If no relevant data is found, return "No relevant numerical data found."
""")

# Define Data Model for Parsing
class ExtractedData(BaseModel):
    datasets: dict[str, list[tuple]] = Field(description="Dictionary of labeled datasets with numerical values.")

# Summarizer Class for Data Extraction
class DataExtractor:
    def __init__(self, file_type):
        self.file_type = file_type
        self.data_parser = PydanticOutputParser(pydantic_object=ExtractedData)

    def get_base_llm(self, streaming: bool = False):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=streaming, api_key=api_key)

    def format_docs_with_ids(self, docs: list[Document]) -> str:
        formatted = [f"Content ID: {doc.metadata['ContentID']}, Source ID: {doc.metadata['ChunkID']}\n{doc.page_content}" for doc in docs]
        return "\n\n" + "\n\n".join(formatted)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
    async def extract_data(self, docs: list[Document]) -> ExtractedData:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_ROLE),
            ("human", "{chain_input}"),
        ])
        llm = self.get_base_llm()
        chain = prompt_template | llm | self.data_parser
        chain_input = textwrap.dedent(f"""
            {DATA_EXTRACTION_PROMPT}
            {self.data_parser.get_format_instructions()}
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

# Function to plot extracted numerical data
def plot_extracted_data(datasets):
    for label, data in datasets.items():
        x_values, y_values = zip(*data)

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label=label)
        plt.title(f"Graph for {label}")
        plt.xlabel("X-axis (independent variable)")
        plt.ylabel("Y-axis (dependent variable)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Path to the PDF file
pdf_path = "ast_sci_data.pdf"

# Initialize the data extractor
data_extractor = DataExtractor(file_type="PDF")

# Extract text from the PDF
document_text = extract_text_from_pdf(pdf_path)

# Convert extracted text into Document objects
documents = [Document(page_content=page, metadata={"ChunkID": idx, "ContentID": None}) for idx, page in enumerate(document_text.split("\n\n"))]

# Run data extraction and generate graphs
async def run_data_extraction():
    extracted_data = await data_extractor.extract_data(documents)

    # Ensure valid data is extracted
    if not extracted_data.datasets or extracted_data.datasets == {"No relevant numerical data found.": []}:
        print("⚠️ No valid numerical data found for graphing.")
        return

    print("✅ Extracted Data:", extracted_data.datasets)  # Debugging: Print extracted numerical datasets

    # Generate graphs from extracted numerical data
    plot_extracted_data(extracted_data.datasets)

# Execute the data extraction
asyncio.run(run_data_extraction())
