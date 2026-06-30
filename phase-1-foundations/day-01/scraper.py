import requests
from bs4 import BeautifulSoup

# ── JUST REPLACE THESE 3 URLS ──────────────────────────────────────────
NEWS_URL     = "https://www.bbc.com/news/articles/c14yn10jzyeo"
SCIENCE_URL  = "https://en.wikipedia.org/wiki/Neural_network"
DIALOGUE_URL = "https://imsdb.com/scripts/Inception.html"
# ───────────────────────────────────────────────────────────────────────

def scrape_paragraphs(url, domain_name, num_paragraphs=5):
    print(f"Scraping {domain_name} from {url}...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        print(f"Failed to fetch {domain_name} — status code {response.status_code}")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # script sites use <pre> tags, news/science use <p> and <li>
    # so we check: if page has <pre> tag, treat it as a script/dialogue page
    pre_tag = soup.find("pre")
    
    if pre_tag:
        # extract raw text from the <pre> block
        raw_script = pre_tag.get_text()
        
        # split into lines and filter blank ones
        lines = [line.strip() for line in raw_script.split("\n") if line.strip()]
        
        # take first num_paragraphs lines as our "paragraphs"
        # increase this number since script lines are short
        selected = lines[:50]
        
        block = f"=== {domain_name.upper()} ===\n" + "\n".join(selected) + "\n\n"
        return block
    
    else:
        # normal news/science page — same logic as before
        elements = soup.find_all(["p", "li"])
        
        clean_elements = []
        for el in elements:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 40:
                if el.name == "li":
                    clean_elements.append("• " + text)
                else:
                    clean_elements.append(text)
        
        selected = clean_elements[:num_paragraphs]
        block = f"=== {domain_name.upper()} ===\n" + "\n\n".join(selected) + "\n\n"
        return block


def scrape_all ():
    corpus = ""
    
    corpus += scrape_paragraphs(NEWS_URL,     domain_name="news",     num_paragraphs=5)
    corpus += scrape_paragraphs(SCIENCE_URL,  domain_name="science",  num_paragraphs=5)
    corpus += scrape_paragraphs(DIALOGUE_URL, domain_name="dialogue", num_paragraphs=5)
    
    # save to input_corpus.txt
    with open("input_corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)
    
    print("Done — input_corpus.txt created successfully.")
    return corpus