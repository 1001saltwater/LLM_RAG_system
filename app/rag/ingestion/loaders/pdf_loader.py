from pathlib import Path
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def load(self, data: bytes) -> list:
        with NamedTemporaryFile(delete=False) as temp:
            temp.write(data)
            temp.flush()
            loader = PyPDFLoader(temp.name)
            return loader.load()
