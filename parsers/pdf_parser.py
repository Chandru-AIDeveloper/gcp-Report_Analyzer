import pdfplumber
from io import BytesIO

def parse_pdf(file_bytes):
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return {"raw_text": text}


