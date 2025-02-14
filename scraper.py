from markdown_it import MarkdownIt
from urllib.parse import urljoin
from validators import url
import requests
import random
from io import BytesIO
from PIL import Image
import hashlib
import time
from aife_time import now_and_choices
from aife_utils import retrieve, RE_NORMALIZE_NEWLINES, RE_REMOVE_MARKDOWN_COMPOSITE_LINKS, RE_REMOVE_MARKDOWN_BASIC_LINKS, RE_REMOVE_HTML_TAGS, RE_REMOVE_INVALID_LINES, RE_COMPRESS_NEWLINES, manage_thread

FIRECRAWL_API_KEYS = [
    retrieve("Firecrawl"),
    retrieve("Firecrawl2"),
    retrieve("Firecrawl3"),
    retrieve("Firecrawl4"),
    retrieve("Firecrawl5"),
    retrieve("Firecrawl6")
]

SPIDER_API_KEY = retrieve("Spider")


def purify(text):
    text = RE_NORMALIZE_NEWLINES.sub("\n", text)
    text = RE_REMOVE_MARKDOWN_COMPOSITE_LINKS.sub("", text)
    text = RE_REMOVE_MARKDOWN_BASIC_LINKS.sub("", text)
    text = RE_REMOVE_HTML_TAGS.sub("", text)
    text = RE_REMOVE_INVALID_LINES.sub("", text)
    text = RE_COMPRESS_NEWLINES.sub("\n", text)
    return text[:50000].strip()


def tidy(text):
    text = RE_NORMALIZE_NEWLINES.sub("\n", text)
    text = RE_REMOVE_INVALID_LINES.sub("", text)
    text = RE_COMPRESS_NEWLINES.sub("\n", text)
    return text[:50000].strip()


def get_lines_and_image_urls(web_url, web_content):
    lines = list(dict.fromkeys(line for line in (line.strip() for line in web_content.splitlines()) if line))
    md = MarkdownIt()
    index = 0
    while index < len(lines):
        items = []
        for child in [child for token in md.parse(lines[index]) if token.type == "inline" for child in token.children]:
            if child.type == "image":
                try:
                    items.append(urljoin(web_url, child.attrs.get("src").lstrip()))
                except Exception:
                    continue
            elif child.type == "text":
                content = child.content.strip()
                if content:
                    items.append(content)
        if items:
            lines[index:index+1] = items
            index += len(items)
        else:
            index += 1
    return dict(enumerate(lines, 1))


def get_lines(web_content):
    lines = list(dict.fromkeys(line for line in (line.strip() for line in web_content.splitlines()) if line))
    return dict(enumerate(lines, 1))


def get_images_and_insert_paths(web_content):
    image_hashes = set()
    for key in list(web_content):
        value = web_content[key]
        if url(value):
            try:
                image = Image.open(BytesIO(requests.get(value, timeout=10).content))
                if max(image.size) < 100:
                    del web_content[key]
                    continue
                if min(image.size) > 1024:
                    ratio = 1024 / min(image.size)
                    image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS)
                image_format = image.format if image.format in ["JPEG", "PNG"] else "JPEG"
                if image_format == "JPEG" and image.mode != "RGB":
                    image = image.convert("RGB")
                image_path = f"temp-images/{now_and_choices()}.{image_format.lower()}"
                buffer = BytesIO()
                image.save(buffer, format=image_format)
                image_data = buffer.getvalue()
                image_hash = hashlib.md5(image_data).hexdigest()
                if image_hash in image_hashes:
                    del web_content[key]
                    continue
                else:
                    image_hashes.add(image_hash)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    web_content[key] = image_path
            except Exception:
                continue
    return web_content


def tidy_web_content(web_raw_content):
    return get_images_and_insert_paths(get_lines(tidy(web_raw_content)))


def tidy_web_contents(web_raw_contents):
    requests = [(tidy_web_content, (web_raw_content,)) for web_raw_content in (web_raw_contents if isinstance(web_raw_contents, list) else [web_raw_contents])]
    return {arguments[0]: result for result, function, arguments in manage_thread(requests)}


def firecrawl(web_url, delay=1):
    url = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": web_url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "actions": [
            {
                "type": "wait",
                "milliseconds": 5000
            },
            {
                "type": "scroll",
                "direction": "down"
            }
        ]
    }
    for attempt, api_key in enumerate(random.sample(FIRECRAWL_API_KEYS, 3), 1):
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, headers=headers).json()
            content = response.get("data", {}).get("markdown")
            if content:
                return content
        except Exception as e:
            print(f"Firecrawl attempt {attempt} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Firecrawl failed to get a valid response after maximum retries")
    return None


def spider(web_url, delay=1):
    url = "https://api.spider.cloud/crawl"
    headers = {
        "Authorization": f"Bearer {SPIDER_API_KEY}",
    }
    json_data = {
        "url": web_url,
        "limit": 1,
        "return_format": "markdown"
    }
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=json_data, timeout=20).json()
            content = response[0].get("content")
            if content:
                return content
        except Exception as e:
            print(f"Spider attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Spider failed to get a valid response after maximum retries")
    return None


def reader(web_url, delay=1):
    url = f"https://r.jina.ai/{web_url}"
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, timeout=20)
            if response.text:
                return response.text
        except Exception as e:
            print(f"Reader attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Reader failed to get a valid response after maximum retries")
    return None


def get_web_content(web_url):
    for index, scraper in enumerate([firecrawl, spider]):
        try:
            web_content = scraper(web_url)
            if web_content:
                if len(web_content) >= 600 or index == len([firecrawl, spider]) - 1:
                    return get_images_and_insert_paths(get_lines_and_image_urls(web_url, tidy(web_content)))
        except Exception:
            continue
    return None


def get_web_contents(web_urls):
    requests = [(get_web_content, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in manage_thread(requests)}


def get_web_text(web_url):
    for index, request in enumerate([reader, spider]):
        try:
            web_text = request(web_url)
            if web_text:
                if len(web_text) >= 600 or index == len([reader, spider]) - 1:
                    return purify(web_text)
        except Exception:
            continue
    return None


def get_web_texts(web_urls):
    requests = [(get_web_text, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in manage_thread(requests)}
