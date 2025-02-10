import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import PyPDF2
import asyncio
import textwrap
from langchain_core.documents.base import Document
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 128000
MAX_DEPTH = 3
AGENT_ROLE = "You're an expert compliance officer, risk manager, auditor and legal advisor"
DEFAULT_BULLET_SUMMARY = textwrap.dedent("""
    Summarize the key points of the document in 3 to 4 bullets, be concise and focus on specific details.
    Only summarize the relevant information to the question otherwise ignore it.
    If there is no relevant information, return "No relevant information found."
""")

class Summary(BaseModel):
    summary: str = Field(description="Summary of the document")

class Summarizer:
    def __init__(self, file_type):
        self.file_type = file_type
        self.summary_parser = PydanticOutputParser(pydantic_object=Summary)

    def get_base_llm(self, streaming: bool = False):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=streaming, api_key=api_key)

    def format_docs_with_ids(self, docs: list[Document]) -> str:
        formatted = [f"Content ID: {doc.metadata['ContentID']}, Source ID: {doc.metadata['ChunkID']}\n{doc.page_content}" for doc in docs]
        return "\n\n" + "\n\n".join(formatted)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
    async def base_summarize(self, docs: list[Document], question: str) -> Summary:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", AGENT_ROLE),
            ("human", "{chain_input}"),
        ])
        llm = self.get_base_llm()
        chain = prompt_template | llm | self.summary_parser
        chain_input = textwrap.dedent(f"""
            Question: {question}
            Please ensure accuracy and brevity with your answers.
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
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Path to the PDF file
document_path = 'PolicyTemplate.pdf'

# Initialize the summarizer for PDF files
summarizer = Summarizer(file_type='PDF')

# Extract text from the PDF
document_text = extract_text_from_pdf(document_path)

# Convert extracted text into Document objects (assuming a simple split for demonstration)
documents = [Document(page_content=page, metadata={"ChunkID": idx, "ContentID": None}) for idx, page in enumerate(document_text.split('\n\n'))]

# Run the summarization
async def run_summarization():
    summary = await summarizer.base_summarize(documents, question="Summarize the document")
    print("Summary:", summary)

# Execute the summarization
asyncio.run(run_summarization())