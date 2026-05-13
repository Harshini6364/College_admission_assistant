# Handles:
# PDF loading
# OCR
# text extraction

import os
import pytesseract

from pdf2image import convert_from_path

from langchain.schema import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from config import POPPLER_PATH


def load_documents(data_folder):

    documents = []

    for file in os.listdir(data_folder):

        file_path = os.path.join(
            data_folder,
            file
        )

        if os.path.isdir(file_path):
            continue

        try:

            if file.endswith(".pdf"):

                loader = PyPDFLoader(file_path)

                loaded_docs = loader.load()

                extracted_text = ""

                for d in loaded_docs:
                    extracted_text += d.page_content.strip()

                # NORMAL PDF
                if len(extracted_text) > 50:

                    documents.extend(loaded_docs)

                # SCANNED PDF
                else:

                    images = convert_from_path(
                        file_path,
                        dpi=150,
                        poppler_path=POPPLER_PATH
                    )

                    ocr_text = ""

                    for img in images:

                        img = img.convert("L")

                        text = pytesseract.image_to_string(img)

                        ocr_text += text + "\n"

                    documents.append(
                        Document(
                            page_content=ocr_text,
                            metadata={"source": file_path}
                        )
                    )

            else:

                loader = TextLoader(file_path)

                documents.extend(loader.load())

        except Exception as e:

            print(f"Could not load {file}: {e}")

    return documents