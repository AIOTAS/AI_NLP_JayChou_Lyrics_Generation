from lxml import etree
import re
import requests
from tqdm import tqdm
import os

def not_contains_chinese(text):
    """判断字符串中不包含中文字符"""
    chinese_regex = re.compile(r'[^\u4e00-\u9fff]')
    return bool(chinese_regex.search(text))


def lyric_spider():
    url = "https://lyrics.net.cn/artist/1"
    html_content = requests.get(url).text
    html = etree.HTML(html_content)
    lyrics_nodes = html.xpath("//a[contains(@href , '/lyrics/')]")
    lyrics_urls_and_names = []

    for node in lyrics_nodes:
        lyric_url = "https://lyrics.net.cn" + node.xpath("./@href")[0]
        lyric_name = node.xpath("./text()")[0]
        lyrics_urls_and_names.append((lyric_url , lyric_name))

    for lyric_url, lyric_name in tqdm(lyrics_urls_and_names):
        lyric_content = requests.get(lyric_url).text
        html = etree.HTML(lyric_content)
        lines = html.xpath("//div[@class='lyrics_main']//div/text()")
        lines = [line.strip() for line in lines]
        with open("lyrics/" + lyric_name.replace("/" , "") + ".txt" , "w" , encoding="utf-8") as fw:
            for line in lines:
                fw.write(line + "\n")


def filter_only_contains_chinese_chars_lyrics():
    base_dir = os.path.abspath("lyrics")
    files = os.listdir(base_dir)

    for f in files:
        file_path = os.path.join(base_dir, f)
        with open(file_path , "rb") as fr:
            bytes = fr.read()
            lines = bytes.decode("utf-8").split("\n")
            lines = [line.strip() for line in lines]

            content = "".join(lines)
            if not not_contains_chinese(content):
                with open("chinese_lyrics/" + f , "wb") as fw:
                    fw.write(bytes)

def merge_chinese_lyrics_to_a_file():
    base_dir = os.path.abspath("chinese_lyrics")
    files = os.listdir(base_dir)

    with open("周杰伦歌词.txt", "w" , encoding="utf-8") as fw:
        for f in files:
            file_path = os.path.join(base_dir, f)
            with open(file_path , "r" , encoding="utf-8") as fr:
                lines = fr.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    fw.write(line + "\n")




if __name__ == "__main__":
    # lyric_spider()
    # filter_only_contains_chinese_chars_lyrics()
    merge_chinese_lyrics_to_a_file()