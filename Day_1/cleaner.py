import re

def clean_text(text):

    # 1. remove citation brackets like [ 1 ], [ 12 ]
    text = re.sub(r'\[\s*\d+\s*\]', ' ', text)

    # 2. remove content inside round brackets like (in Japanese), (in English)
    text = re.sub(r'\(.*?\)', ' ', text)
  

    # 3. remove bullet symbols
    text = re.sub(r'•', ' ', text)
    

    # 4. remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # 5. remove punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
   

    # 6. remove extra whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. lowercase everything
    text = text.lower()

    return text
