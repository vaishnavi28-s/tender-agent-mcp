# fetch.py

import os
import re
import time
import json
import requests
import feedparser
import subprocess
from bs4 import BeautifulSoup
from pydantic import BaseModel
from urllib.parse import urlencode, urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

SEARCH_TERM = "wahlunterlagen"
BASE_URL = "https://www.service.bund.de/Content/DE/Ausschreibungen/Suche/Formular.html"
HEADERS = {"User-Agent": "TenderBot/1.0"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_DIR = "tender_vector_db"

class Tender(BaseModel):
    title: str
    link: str
    deadline: str
    announcement_url: str | None
    city: str | None = None
    md_file: str | None = None

def get_rss_url_from_search(search_term: str) -> str:
    params = {
        "nn": "4641482",
        "type": "0",
        "resultsPerPage": "100",
        "templateQueryString": search_term,
        "sortOrder": "dateOfIssue_dt desc",
        "jobsrss": "true"
    }
    return f"{BASE_URL}?{urlencode(params)}"

def fetch_rss_entries():
    rss_url = get_rss_url_from_search(SEARCH_TERM)
    r = requests.get(rss_url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    return [{"title": e.title, "link": e.link} for e in feed.entries]

def extract_deadline_and_announcement(html: str):
    soup = BeautifulSoup(html, "html.parser")
    deadline = "—"
    dt = soup.find("dt", string=lambda t: t and "Angebotsfrist" in t)
    if dt:
        dd = dt.find_next_sibling("dd")
        if dd:
            deadline = dd.get_text(strip=True)

    a = soup.find("a", string=lambda t: t and "Bekanntmachung" in t)
    href = None
    if a:
        href = a.get("href")
        if href:
            href = " ".join(href.split())
            if not href.startswith("http"):
                href = "https://www.service.bund.de" + href

    return deadline, href if href else None

def extract_city(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    dt = soup.find("dt", string=lambda t: t and "Erfüllungsort" in t)
    if dt:
        dd = dt.find_next_sibling("dd")
        if dd:
            city_text = dd.get_text(strip=True)
            if city_text:
                return city_text.split(",")[0].strip()
    return None

def extract_all_tab_urls(base_url):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    driver.get(base_url)
    time.sleep(3)

    base = urlparse(base_url)
    root = f"{base.scheme}://{base.netloc}"
    tab_urls = set()

    for link in driver.find_elements(By.TAG_NAME, "a"):
        href = link.get_attribute("href")
        if href and "/VMPSatellite/public/company/project/" in href and "/de/" in href:
            clean = re.sub(r";jsessionid=[^?#]*", "", href)
            full_url = urljoin(root, clean)
            tab_urls.add(full_url)

    driver.quit()
    return sorted(tab_urls)

def run_crwl(url):
    try:
        result = subprocess.run(["crwl", "crawl", url, "-o", "markdown"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return ""

def clean_all_markdown_files(directory="."):
    noisy_patterns = [
        r'^\s*[*\-_=]{3,}\s*$',
        r'\[.*\]\(javascript:.*\)',
        r'\[.*\]\(#.*\)',
        r'\[.*\]\(\)',
        r'^Bitte warten.*',
        r'^\s*\[\s*\]\s*$',
    ]
    noisy_keywords = [
        "impressum", "datenschutz", "barrierefreiheit", "systemzeit",
        "administration intelligence", "cosinex", "d-nrw", "vo:",
        "vmp", "zurück", "anmelden", "teilnehmen", "seite drucken",
        "javascript", "bitte warten", "mandantennummer"
    ]

    def is_noisy(line):
        l = line.lower()
        return any(k in l for k in noisy_keywords) or any(re.search(p, line) for p in noisy_patterns)

    for fname in os.listdir(directory):
        if fname.endswith(".md"):
            with open(fname, "r", encoding="utf-8") as f:
                lines = f.readlines()
            cleaned = [line for line in lines if not is_noisy(line)]
            with open(fname, "w", encoding="utf-8") as f:
                f.writelines(cleaned)

def fetch_and_process():
    existing = []
    if os.path.exists("tenders_metadata.json"):
        with open("tenders_metadata.json", "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_links = {t["link"] for t in existing}
    fetched = fetch_rss_entries()

    for entry in fetched:
        if entry["link"] in existing_links:
            continue

        try:
            html = requests.get(entry["link"], headers=HEADERS, timeout=10).text
            deadline, ann_url = extract_deadline_and_announcement(html)
            city = extract_city(html)

            tender = Tender(
                title=entry["title"],
                link=entry["link"],
                deadline=deadline,
                announcement_url=ann_url,
                city=city
            ).model_dump()

            if ann_url:
                tab_links = extract_all_tab_urls(ann_url)
                if ann_url not in tab_links:
                    tab_links.insert(0, ann_url)

                full_md = ""
                for url in tab_links:
                    content = run_crwl(url)
                    if content:
                        full_md += f"\n\n# Content from {url}\n\n{content}\n{'=' * 80}\n"

                safe_title = re.sub(r"[^\w\s-]", "", entry["title"]).strip().replace(" ", "_")[:60]
                md_filename = f"{safe_title}.md"
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write(full_md)
                tender["md_file"] = md_filename

            existing.append(tender)

        except Exception as e:
            print(f"Error processing tender: {e}")

    with open("tenders_metadata.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    with open("tenders_index.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    clean_all_markdown_files()

def build_vector_store():
    documents = []
    with open("tenders_index.json", "r", encoding="utf-8") as f:
        index_data = json.load(f)
    md_map = {t["md_file"]: t for t in index_data if "md_file" in t}

    for file in os.listdir("."):
        if file.endswith(".md"):
            with open(file, encoding="utf-8") as f:
                text = f.read()
            meta = md_map.get(file, {})
            documents.append(Document(page_content=text, metadata={"source": file, **meta}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    vectordb.persist()

if __name__ == "__main__":
    fetch_and_process()
    build_vector_store()
