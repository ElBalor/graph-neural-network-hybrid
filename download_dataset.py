"""Download Big-Vul dataset with retry logic."""
import os
import sys
import time
import requests
from pathlib import Path

URL = "https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/master/MSR_data_cleaned.csv"
OUTPUT_PATH = Path("data/MSR_data_cleaned.csv")

def download_with_retry(url, output_path, max_retries=3, timeout=300):
    """Download file with retry logic."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Big-Vul dataset from {url}...")
    print(f"Output: {output_path}")
    print("This may take several minutes (file is ~100MB+)...\n")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
            
            print(f"\n✓ Download complete! File saved to {output_path}")
            
            # Verify it's actually CSV, not HTML
            with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('<!DOCTYPE') or first_line.startswith('<html'):
                    print("✗ Warning: Downloaded file appears to be HTML, not CSV!")
                    output_path.unlink()
                    raise ValueError("Downloaded HTML instead of CSV")
                else:
                    print(f"✓ Verified: File appears to be valid CSV (first line: {first_line[:50]}...)")
            
            return True
            
        except requests.exceptions.Timeout:
            print(f"✗ Timeout on attempt {attempt}")
            if attempt < max_retries:
                wait_time = attempt * 10
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                wait_time = attempt * 10
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    print(f"\n✗ Failed to download after {max_retries} attempts.")
    print("\nAlternative: Download manually from:")
    print("https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset")
    print(f"Save it as: {output_path.absolute()}")
    return False

if __name__ == "__main__":
    success = download_with_retry(URL, OUTPUT_PATH)
    sys.exit(0 if success else 1)

