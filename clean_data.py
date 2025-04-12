import os
import re

def clean_gutenberg_text(text):
    # Remove Gutenberg header
    start_match = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", text)
    if start_match:
        text = text[start_match.end():]

    # Remove Gutenberg footer
    end_match = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", text)
    if end_match:
        text = text[:end_match.start()]

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove non-ASCII characters (optional)
    text = text.encode('ascii', errors='ignore').decode()

    # Strip leading/trailing spaces
    return text.strip()

def process_books(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            cleaned_text = clean_gutenberg_text(raw_text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            print(f"✔️ Cleaned: {filename}")

# Run the cleaner
process_books("data", "data_clean")
