from app.rag.ingestion.preprocessor import TextPreprocessor


def test_preprocess():

    pre = TextPreprocessor()

    text = "Hello\r\n\r\nWorld"

    result = pre.preprocess(text)

    assert result == "Hello\n\nWorld"