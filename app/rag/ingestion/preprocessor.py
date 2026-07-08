import re


class TextPreprocessor:

    def preprocess(self, text:str):

        # 统一换行
        text=text.replace("\r\n","\n")
        text=text.replace("\r","\n")

        # 删除隐藏字符
        text=text.replace("\u200b","")

        # 删除首尾
        text=text.strip()


        # 保留代码结构，只删除普通行尾空格
        lines=[line.rstrip()for line in text.splitlines()]

        text="\n".join(lines)

        # markdown标题规范化
        text=re.sub(r"^(#+)\s*",r"\1 ",text,flags=re.MULTILINE)


        # 删除markdown分隔符
        text=re.sub(r"^[-*_]{3,}$","",text,flags=re.MULTILINE)


        # markdown链接转文字
        text=re.sub(r"\[(.*?)\]\(.*?\)",r"\1",text)


        # 多余空格
        text=re.sub(r"[ \t]+"," ",text)


        # 空行控制
        text=re.sub(r"\n{3,}","\n\n",text)


        return text.strip()