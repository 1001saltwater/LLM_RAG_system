from app.rag.ingestion.splitter import TextSplitter



def test_splitter():

    text = """
FastAPI 是一个现代化的 Python Web 框架，它以高性能和简洁的开发体验著称。开发者可以利用类型提示自动完成参数校验、接口文档生成以及数据序列化，因此在 AI 服务开发中被广泛采用。相比传统框架，它能够减少大量样板代码，使开发者更加关注业务逻辑本身。

在一个典型的 RAG 系统中，请求首先到达 FastAPI 提供的 HTTP 接口。接口负责接收用户上传的文档，然后将文档交给预处理模块进行清洗。预处理模块会去除多余的空格、统一换行符、删除不可见字符，并保证文本具有统一的格式。这一步虽然简单，却直接影响后续切分和向量化的效果，因此通常作为整个 Pipeline 的第一步。

文本经过预处理之后，会进入 Splitter。Splitter 的职责不是计算向量，也不是访问数据库，而是把一篇长文档切分成多个适合 Embedding 模型处理的小块。如果一个 Chunk 过长，Embedding 模型可能无法完整编码；如果 Chunk 太短，又可能丢失上下文。因此，一个好的切分策略需要在语义完整性和长度限制之间取得平衡。

目前我们的第一版 Splitter 采用按段落切分的策略。程序首先根据连续空行划分段落，然后尝试把多个较短段落合并成一个 Chunk。当继续加入下一段会超过最大长度限制时，就结束当前 Chunk，并开始构建新的 Chunk。这样的实现虽然简单，但是已经能够满足绝大多数学习阶段的需求，也方便以后逐步扩展到按句子、按 Token 甚至按语义进行切分。

完成切分之后，每一个 Chunk 都会进入 Embedding 模块。Embedding 模型负责把自然语言转换为高维向量，这些向量随后会保存到向量数据库中。当用户提出问题时，系统会首先把问题编码成向量，再利用向量相似度检索出最相关的几个 Chunk，最后将这些 Chunk 与用户的问题一起发送给大语言模型生成回答。这就是一个典型 RAG 系统的基本工作流程。
"""

    splitter = TextSplitter(max_length=500)

    chunks = splitter.split(text)

    print("\n================ Split Result ================\n")

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}")
        print(f"Length : {len(chunk)}")
        print("-" * 60)
        print(chunk)
        print("=" * 60)

    # 基本断言
    assert len(chunks) > 1

    # 每个 Chunk 都应该有内容
    for chunk in chunks:
        assert len(chunk.strip()) > 0