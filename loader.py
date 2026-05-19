# Handles:
# PDF loading (text, scanned, tables, images)
# Text file loading
# Uses Docling for robust PDF extraction

import os

from langchain.schema import Document

from langchain_community.document_loaders import TextLoader

from docling.document_converter import DocumentConverter


# Single converter instance reused across all files
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
    loader = TextLoader(file_path)
    return loader.load()


def load_documents(data_folder: str) -> list[Document]:
    """
    Load all documents from the given folder.
    Supports: PDF (text/scanned/tables/images), TXT
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

            else:
                docs = _load_text(file_path)
                documents.extend(docs)

        except Exception as e:
            print(f"Could not load {file}: {e}")

    return documents