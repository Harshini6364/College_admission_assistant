# all required paths and constants

import pytesseract

# OCR PATH
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# POPPLER PATH
POPPLER_PATH = (
    r"C:\Users\BHASHINI\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
)

# EMBEDDING MODEL
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LLM MODEL
LLM_MODEL = "llama-3.1-8b-instant"