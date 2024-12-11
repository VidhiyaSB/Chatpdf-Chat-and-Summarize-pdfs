import PyPDF2
from typing import List

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks