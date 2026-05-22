# ===========================================
# ABSTRACTION LAYER: Document Loading
# GOF Pattern: Factory Method
# Responsibility: Load and parse documents
# Input: folder path
# Output: list[Document]
# ===========================================

import os

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from docling.document_converter import DocumentConverter

from utils.logger import get_logger

logger = get_logger(__name__)

# Single Docling converter instance reused across all files (Singleton-like)
_converter = DocumentConverter()


# ---------------------------
# GOF FACTORY METHOD PATTERN
# ---------------------------

class DocumentLoaderFactory:
    """
    GOF Factory Method Pattern:
    Creates the correct loader based on file type.
    Adding support for a new file type only requires
    adding one entry here — nothing else changes.

    Low Coupling: loader.py does not know about rag.py or app.py
    High Cohesion: only responsible for deciding which loader to use
    """

    @classmethod
    def get_loader(cls, file_path: str):
        """
        Return the correct loader callable for the given file.
        Raises ValueError for unsupported file types.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pdf":
            return cls._load_pdf
        elif ext == ".txt":
            return cls._load_txt
        else:
            raise ValueError(f"No loader registered for file type: {ext}")

    @staticmethod
    def _load_pdf(file_path: str) -> list[Document]:
        """
        Load PDF using Docling.
        Handles text, scanned, tables, and image-based PDFs.
        Falls back to PyPDFLoader if Docling fails (e.g. low RAM).
        """
        try:
            result = _converter.convert(file_path)

            if result.status.name == "FAILURE":
                raise RuntimeError(f"Docling conversion failed: {result.errors}")

            markdown_text = result.document.export_to_markdown()

            logger.info(f"Docling loaded: {os.path.basename(file_path)}")

            return [Document(
                page_content=markdown_text,
                metadata={"source": file_path}
            )]

        except Exception as e:
            logger.warning(f"Docling failed for {os.path.basename(file_path)}: {e}")
            logger.info(f"Falling back to PyPDFLoader for {os.path.basename(file_path)}")

            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            logger.info(f"PyPDFLoader loaded: {os.path.basename(file_path)} — {len(docs)} page(s)")
            return docs

    @staticmethod
    def _load_txt(file_path: str) -> list[Document]:
        """
        Load plain text file using LangChain TextLoader.
        """
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"TextLoader loaded: {os.path.basename(file_path)}")
        return docs


# ---------------------------
# LOAD FUNCTIONS
# ---------------------------

def _load_file(file_path: str) -> list[Document]:
    """
    Load a single file using the Factory.
    Returns list of Documents.
    """
    loader_fn = DocumentLoaderFactory.get_loader(file_path)
    return loader_fn(file_path)


def load_documents(data_folder: str) -> list[Document]:
    """
    Load all supported documents from the given folder.
    Skips unsupported file types with a warning.

    Single Responsibility: only loads, does not chunk or embed.
    """
    documents = []

    if not os.path.exists(data_folder):
        logger.error(f"Data folder not found: {data_folder}")
        return documents

    for file in os.listdir(data_folder):

        file_path = os.path.join(data_folder, file)

        if os.path.isdir(file_path):
            continue

        try:
            docs = _load_file(file_path)
            documents.extend(docs)

        except ValueError as e:
            logger.warning(f"Skipping {file}: {e}")

        except Exception as e:
            logger.error(f"Could not load {file}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents