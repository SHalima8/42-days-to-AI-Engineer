import requests
from bs4 import BeautifulSoup

NEWS_URL     = "https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake"
SCIENCE_URL  = "https://en.wikipedia.org/wiki/Artificial_neural_network"
DIALOGUE_URL = "https://en.wikipedia.org/wiki/Dialogue"

def scrape_paragraphs(url, domain_name, num_paragraphs=25):
    print(f"Scraping {domain_name} from {url}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        print(f"Failed to fetch {domain_name} — status code {response.status_code}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # only get <p> tags — skip <li> entirely
    # this removes all table of contents and navigation bullets
    elements = soup.find_all("p")

    clean_elements = []
    for el in elements:
        text = el.get_text(separator=" ", strip=True)

        # skip short paragraphs — likely captions or empty tags
        if len(text) < 80:
            continue

        # skip paragraphs that look like references or citations only
        if text.startswith("[") or text.count("[") > 5:
            continue

        clean_elements.append(text)

    selected = clean_elements[:num_paragraphs]
    block = f"=== {domain_name.upper()} ===\n" + "\n".join(selected) + "\n\n"
    return block


def scrape_all():
    corpus = ""
    corpus += scrape_paragraphs(NEWS_URL,     domain_name="news",     num_paragraphs=25)
    corpus += scrape_paragraphs(SCIENCE_URL,  domain_name="science",  num_paragraphs=25)
    corpus += scrape_paragraphs(DIALOGUE_URL, domain_name="dialogue", num_paragraphs=25)

    with open("input_corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    print("Done — input_corpus.txt created successfully.")
    return corpus


if __name__ == "__main__":
    scrape_all()