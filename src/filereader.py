import os
import pdfplumber
from docx import Document
import pandas as pd
import pytesseract

class FileReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.extension = os.path.splitext(filepath)[1].lower()

    def read(self):
        if self.extension == ".pdf":
            return self.read_pdf()
        if self.extension == ".docx":
            return self.read_word()
        if self.extension in [".xls", ".xlsx"]:
            return self.read_excel()
        if self.extension == ".csv":
            return self.read_csv()
        if self.extension == ".txt":
            return self.read_txt()
        return self.read_txt()

    def read_pdf(self):
        text_parts = []
        try:
            with pdfplumber.open(self.filepath) as pdf:
                for page in pdf.pages:
                    try:
                        content = page.extract_text()
                        if content and content.strip():
                            lines = [l.strip() for l in content.splitlines() if l.strip()]
                            text_parts.append("\n".join(lines))
                        else:
                            img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(img)
                            if ocr_text and ocr_text.strip():
                                lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
                                text_parts.append("\n".join(lines))
                    except Exception:
                        continue
        except Exception as e:
            print(f"[FileReader] Error reading PDF {self.filepath}: {e}")
        return "\n".join(text_parts).strip()

    def read_word(self):
        try:
            doc = Document(self.filepath)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            print(f"[FileReader] Error reading Word file {self.filepath}: {e}")
            return ""

    def read_excel(self):
        try:
            df = pd.read_excel(self.filepath, dtype=str).fillna("")
            rows = df.agg(" ".join, axis=1).tolist()
            return "\n\n".join(rows)
        except Exception as e:
            print(f"[FileReader] Error reading Excel {self.filepath}: {e}")
            return ""

    def read_csv(self):
        try:
            df = pd.read_csv(self.filepath, dtype=str).fillna("")
            rows = df.agg(" ".join, axis=1).tolist()
            return "\n\n".join(rows)
        except Exception as e:
            print(f"[FileReader] Error reading CSV {self.filepath}: {e}")
            return ""

    def read_txt(self):
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                return "\n".join(lines)
        except Exception as e:
            print(f"[FileReader] Error reading TXT {self.filepath}: {e}")
            return ""
