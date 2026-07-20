from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.config import settings

class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    
    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)