
import urllib.request
import os

# List of books to download (approx 10MB total)
# IDs: War and Peace (2600), Moby Dick (2701), Frankenstein (84), Dracula (345), 
# Pride and Prejudice (1342), Ulysses (4300), The Count of Monte Cristo (1184)
books = [
    {"url": "https://www.gutenberg.org/files/1661/1661-0.txt", "filename": "sherlock.txt"},
    {"url": "https://www.gutenberg.org/files/2600/2600-0.txt", "filename": "war_and_peace.txt"},
    {"url": "https://www.gutenberg.org/files/2701/2701-0.txt", "filename": "moby_dick.txt"},
    {"url": "https://www.gutenberg.org/files/84/84-0.txt", "filename": "frankenstein.txt"},
    {"url": "https://www.gutenberg.org/cache/epub/345/pg345.txt", "filename": "dracula.txt"},
    {"url": "https://www.gutenberg.org/files/1342/1342-0.txt", "filename": "pride_and_prejudice.txt"},
    {"url": "https://www.gutenberg.org/files/1184/1184-0.txt", "filename": "monte_cristo.txt"}
]

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"Downloading approximately {len(books)} books to {data_dir}...")

for book in books:
    url = book["url"]
    filename = book["filename"]
    file_path = os.path.join(data_dir, filename)
    
    print(f"Downloading {filename} from {url}...")
    try:
        # User-Agent header might be needed for some servers, but Gutenberg usually allows direct access
        # Adding a simple User-Agent just in case
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        )
        
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8', errors='ignore')
            # Basic cleanup: Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("Download complete.")
