import pdfplumber
from PIL import Image
import pytesseract
from docx import Document
import os
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class FileReader:

    def __init__(self, filepath):
        self.filepath = filepath
        self.extension = os.path.splitext(filepath)[-1].lower()

    def read(self):
        if self.extension == ".pdf":
            return self.read_pdf()
        elif self.extension == ".docx":
            return self.read_word()
        elif self.extension in [".xls", ".xlsx"]:
            return self.read_excel()
        elif self.extension == ".txt":
            return self.read_txt()
        elif self.extension == ".csv":
            return self.read_csv()
        else:
            raise ValueError(f"Unsupported file type: {self.extension}")

    def read_pdf(self):
        text = ""
        try:
            with pdfplumber.open(self.filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    content = page.extract_text()
                    if content:
                        lines = [line.strip() for line in content.splitlines() if line.strip()]
                        text += "\n".join(lines) + "\n"
                    else:
                        img = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text:
                            lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
                            text += "\n".join(lines) + "\n"
        except Exception as e:
            print(f"Error reading PDF '{self.filepath}': {e}")
        return text

    def read_word(self):
        try:
            doc = Document(self.filepath)
            lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(lines)
        except Exception as e:
            print(f"Error reading Word file '{self.filepath}': {e}")
            return ""

    def read_excel(self):
        try:
            df = pd.read_excel(self.filepath)
            content = " ".join(
                " ".join(df[col].dropna().astype(str)) for col in df.columns
            )
            return content
        except Exception as e:
            print(f"Error reading Excel '{self.filepath}': {e}")
            return ""

    def read_csv(self):
        try:
            df = pd.read_csv(self.filepath)

            if df.shape[1] == 1:
                content = " ".join(df.iloc[:, 0].dropna().astype(str))
            else:
                content = " ".join(
                    " ".join(df[col].dropna().astype(str)) for col in df.columns
                )
            return content

        except Exception as e:
            print(f"Error reading CSV '{self.filepath}': {e}")
            return ""

    def read_txt(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                return "\n".join(lines)
        except Exception as e:
            print(f"Error reading TXT '{self.filepath}': {e}")
            return ""