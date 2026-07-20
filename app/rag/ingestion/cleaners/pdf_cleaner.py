import re
from langchain_core.documents import Document

class PDFCleaner:
    def __init__(self):
        self.pdf_loader = PDFLoader()

    def clean(self, documents: list) -> list:

        cleaned_documents = []
        for document in documents:
            text = document.page_content
            text = self.normalize_withspaces(text)
            text = self.remove_empty_lines(text)
            text = self.fix_line_breaks(text)
            text = self.remove_headers(text)
            if text.strip():
                cleaned_documents.append(Document(page_content=text,metadata=document.metadata))
        return cleaned_documents

    def normalize_withspaces(self, text: str) -> str:
        return re.sub(r'[\t]+', ' ', text)

    def remove_empty_lines(self, text: str) -> str:
        return re.sub(r'\n{3,}', '\n\n', text)

    def fix_line_breaks(self, text: str) -> str:
        text = text.replace('-\n', '')
        text = text.replace('\n', ' ')
        return text

    def remove_headers(self, text: str) -> str:
        patterns = [r'第\d+页', r'Page\s\d+']
        for pattern in patterns: 
            text = re.sub(pattern, '', text)
        return text
