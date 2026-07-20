from app.rag.ingestion.loaders.pdf_loader import PDFLoader
from app.rag.ingestion.cleaners.pdf_cleaner import PDFCleaner
from app.rag.splitter.LangChain_splitter import TextSplitter

class IngestionPipeline:
    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.pdf_cleaner = PDFCleaner()
        self.text_splitter = TextSplitter()

    def ingest(self, data: bytes) -> list[Document]:
        documents = self.pdf_loader.load(data)
        documents = self.pdf_cleaner.clean(documents)
        chunks = self.text_splitter.split(documents)
        return chunks


