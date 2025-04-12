from transformers import GPT2Tokenizer
from datasets import load_dataset, Dataset
import os

def load_all_texts(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n\n"
    return all_text

# Load and chunk
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

if __name__ == "__main__":
    data_path = "./data_clean"
    raw_text = load_all_texts(data_path)
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Needed for GPT2

    # Chunk into list of strings
    chunks = chunk_text(raw_text, chunk_size=1000)

    # Wrap in Dataset
    dataset = Dataset.from_dict({"text": chunks})
    dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

    dataset.save_to_disk("kipling_dataset")
    print("Dataset saved as kipling_dataset")
