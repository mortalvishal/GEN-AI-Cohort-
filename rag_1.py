from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = Path(__file__).parent / "notes.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_doc = text_splitter.split_documents(documents=docs)

print("DOCS", len(docs))
print("SPLIT",len(split_doc))
