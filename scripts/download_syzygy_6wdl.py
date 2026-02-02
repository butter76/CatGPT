#!/usr/bin/env python3
"""
Script to download all Syzygy 6-WDL tablebase files from tablebase.sesse.net
"""

import re
import subprocess
import sys
import os
import requests
from pathlib import Path
import time

# Base URL and target directory
BASE_URL = "http://tablebase.sesse.net/syzygy/6-WDL/"
TARGET_DIR = "syzygy_tables/6-WDL"

def fetch_directory_listing(url):
    """Fetch the HTML directory listing from the given URL."""
    print(f"Fetching directory listing from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching directory listing: {e}")
        sys.exit(1)

def extract_filenames(html_content):
    """Extract all .rtbw filenames from the Apache directory listing HTML."""
    # Pattern to match .rtbw files in Apache directory listing
    # Look for links with .rtbw extension
    pattern = r'<a href="([^"]+\.rtbw)">'
    filenames = re.findall(pattern, html_content, re.IGNORECASE)

    # Alternative pattern in case the first one doesn't work
    if not filenames:
        # Look for .rtbw files in a more general way
        pattern = r'([A-Z][A-Za-z0-9]+\.rtbw)'
        filenames = re.findall(pattern, html_content)
        # Remove duplicates while preserving order
        filenames = list(dict.fromkeys(filenames))

    return sorted(filenames)

def create_target_directory(target_dir):
    """Create the target directory if it doesn't exist."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")

def download_file(filename, base_url, target_dir):
    """Download a single file using wget."""
    file_url = f"{base_url}{filename}"
    target_path = os.path.join(target_dir, filename)

    # Skip if file already exists and has reasonable size
    if os.path.exists(target_path):
        file_size = os.path.getsize(target_path)
        if file_size > 1000:  # If file is larger than 1KB, assume it's complete
            print(f"✓ Skipping {filename} (already exists, {file_size} bytes)")
            return True

    print(f"⬇ Downloading {filename}...")

    try:
        # Use wget with useful options
        cmd = [
            'wget',
            '--no-verbose',           # Less output
            '--continue',             # Resume partial downloads
            '--timeout=300',          # 5 minute timeout
            '--tries=3',              # Retry up to 3 times
            '--user-agent=Mozilla/5.0 (Linux; Syzygy Tablebase Downloader)',
            '--output-document=' + target_path,
            file_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            file_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
            print(f"✓ Downloaded {filename} ({file_size} bytes)")
            return True
        else:
            print(f"✗ Failed to download {filename}: {result.stderr.strip()}")
            return False

    except subprocess.SubprocessError as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {filename}: {e}")
        return False

def main():
    """Main function to orchestrate the download process."""
    print("=== Syzygy 6-WDL Tablebase Downloader ===")
    print()

    # Create target directory
    create_target_directory(TARGET_DIR)

    # Fetch and parse directory listing
    html_content = fetch_directory_listing(BASE_URL)
    filenames = extract_filenames(html_content)

    if not filenames:
        print("No .rtbw files found in the directory listing!")
        sys.exit(1)

    print(f"Found {len(filenames)} tablebase files to download:")
    for i, filename in enumerate(filenames[:10]):  # Show first 10
        print(f"  {i+1:3d}. {filename}")
    if len(filenames) > 10:
        print(f"  ... and {len(filenames) - 10} more")
    print()

    # Confirm with user
    response = input(f"Download all {len(filenames)} files to {TARGET_DIR}? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        sys.exit(0)

    # Download files
    print(f"\nStarting downloads...")
    print("=" * 50)

    successful = 0
    failed = 0

    for i, filename in enumerate(filenames, 1):
        print(f"\n[{i}/{len(filenames)}] ", end="")

        if download_file(filename, BASE_URL, TARGET_DIR):
            successful += 1
        else:
            failed += 1

        # Small delay to be nice to the server
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 50)
    print(f"Download completed!")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📁 Files saved to: {TARGET_DIR}")

if __name__ == "__main__":
    main()
