from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore 

pdf_path = Path(__file__).parent / "notes.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_doc = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key= "OPENAI_API_KEY",
)
# this is for the injection part

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder,
# )

# print("DOCS", len(docs))
# print("SPLIT",len(split_doc))

# vector_store.add_documents(documents=split_doc)
print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder,
)

# search_result = retriver.similarity_search(
#     query="What is FS Module?"
# )

# Print("Relevant Chunks", search_result)

relevant_chunks = retriver.similarity_search(
    query="What is FS Module?"
)

SYSTEM_PROMPT = """
You are an helpful AI Assistant who responds base of the available context.

Context:
{relevant_chunks}
"""