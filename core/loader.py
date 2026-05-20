import os
 
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from docling.document_converter import DocumentConverter
 
from utils.logger import get_logger
 
logger = get_logger(__name__)
 
_converter = DocumentConverter()
 
 
def _load_pdf(file_path: str) -> Document:
    """
    Load a PDF using Docling.
    Handles text, scanned, tables, and image-based PDFs.
    Returns a single Document with markdown-formatted content.
    """
    result = _converter.convert(file_path)
    markdown_text = result.document.export_to_markdown()
 
    return Document(
        page_content=markdown_text,
        metadata={"source": file_path}
    )
 
 
def _load_text(file_path: str) -> list[Document]:
    """
    Load a plain text file using LangChain's TextLoader.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
 
 
def load_documents(data_folder: str) -> list[Document]:
    """
    Load all documents from the given folder.
    Supports: PDF (text / scanned / tables / images), TXT.
    """
    documents = []
 
    for file in os.listdir(data_folder):
 
        file_path = os.path.join(data_folder, file)
 
        if os.path.isdir(file_path):
            continue
 
        try:
            if file.endswith(".pdf"):
                doc = _load_pdf(file_path)
                documents.append(doc)
                logger.info(f"Loaded PDF: {file}")
 
            else:
                docs = _load_text(file_path)
                documents.extend(docs)
                logger.info(f"Loaded text file: {file}")
 
        except Exception as e:
            logger.error(f"Could not load {file}: {e}")
 
    return documents
 
