
import urllib.request
import os

url = "https://www.gutenberg.org/files/1661/1661-0.txt"
data_dir = "data"
file_path = os.path.join(data_dir, "sherlock.txt")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"Downloading {url}...")
try:
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
        # Basic cleanup: Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"Successfully saved to {file_path}")
except Exception as e:
    print(f"Error downloading: {e}")
