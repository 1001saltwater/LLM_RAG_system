from app.config.config import settings

class TextSplitter:
    """
    按段落切分文本。

    多个较短段落会自动合并，
    保证每个 Chunk 尽可能接近 max_length，
    同时不会截断段落。
    """

    def __init__(self, max_length: settings.CHUNK_SIZE, overlap: settings.CHUNK_OVERLAP):
        self.max_length = max_length
        self.overlap = overlap

    def split(self, text: str) -> list[str]:

        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # 当前 Chunk 为空，直接加入
            if not current_chunk:
                current_chunk = paragraph
                continue

            # 尝试把下一段加入当前 Chunk
            candidate = current_chunk + "\n\n" + paragraph

            if len(candidate) <= self.max_length:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph

        # 最后一块不要忘记保存
        if current_chunk:
            chunks.append(current_chunk)

        return chunks