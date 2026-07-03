import re

class TextPreprocessor:
    def preprocess(self, test: str):
        
        # 替换换行符
        text = test.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        
        # 替换全角空格
        text = text.replace("\u200b", "")

        # 移除空行
        lines = [line.strip() for line in text.splitlines("\n")]
        text = "\n".join(lines)

        # 合并多个换行符
        # 移除连续3个以上的空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 移除首尾空格
        text = text.strip()

        return text