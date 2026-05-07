import json
import os
from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv(dotenv_path="./.env")

# ----------------------------
# Step 1: Configure Docling
# ----------------------------
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

pdf_path = Path("./lab3/sample_pdfs/health_policy.pdf")

# ----------------------------
# Step 2: Convert PDF
# ----------------------------
print(f"📃 Processing: {pdf_path.name}")

result = converter.convert(str(pdf_path))
doc = result.document

markdown_output = doc.export_to_markdown()

print("\n=== Markdown Output ===\n")
print(markdown_output)

# Save markdown
os.makedirs("./lab3/results", exist_ok=True)
with open("./lab3/results/extracted_policy.md", "w") as f:
    f.write(markdown_output)

print("\n✅Saved Markdown to extracted_policy.md")

# ----------------------------
# Step 3: Extract Structured Data
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract structured data from this insurance policy. "
     "Return JSON with keys: policy_name, sum_insured, exclusions[], benefits[], waiting_periods[]. "
     "Only return JSON."),
    ("human", "Policy Document:\n{document}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"document": markdown_output})

parsed = json.loads(response)

print("\n=== Extracted JSON ===\n")
print(json.dumps(parsed, indent=2))