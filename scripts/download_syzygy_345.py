#!/usr/bin/env python3
"""
Script to download all Syzygy 3-4-5 tablebase files from tablebase.sesse.net
Supports both WDL (.rtbw) and DTZ (.rtbz) files
"""

import re
import subprocess
import sys
import os
import requests
from pathlib import Path
import time
import argparse

# Base URL and target directory
BASE_URL = "http://tablebase.sesse.net/syzygy/3-4-5/"
TARGET_DIR = "syzygy_tables/3-4-5"

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

def extract_filenames(html_content, file_types=None):
    """Extract tablebase filenames from the Apache directory listing HTML."""
    if file_types is None:
        file_types = ['rtbw', 'rtbz']

    all_filenames = []

    for ext in file_types:
        # Pattern to match tablebase files in Apache directory listing
        # Look for links with the specified extension
        pattern = rf'<a href="([^"]+\.{ext})">'
        filenames = re.findall(pattern, html_content, re.IGNORECASE)

        # Alternative pattern in case the first one doesn't work
        if not filenames:
            # Look for files in a more general way
            pattern = rf'([A-Z][A-Za-z0-9]+\.{ext})'
            filenames = re.findall(pattern, html_content)

        all_filenames.extend(filenames)

    # Remove duplicates while preserving order, then sort
    filenames = list(dict.fromkeys(all_filenames))
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

def get_file_type_choice():
    """Get user's choice for which file types to download."""
    print("Syzygy 3-4-5 contains two types of tablebase files:")
    print("  1. WDL files (.rtbw) - Win/Draw/Loss information")
    print("  2. DTZ files (.rtbz) - Distance to Zero information")
    print("  3. Both WDL and DTZ files")
    print()

    while True:
        choice = input("Which files would you like to download? [1/2/3]: ").strip()
        if choice == '1':
            return ['rtbw']
        elif choice == '2':
            return ['rtbz']
        elif choice == '3':
            return ['rtbw', 'rtbz']
        else:
            print("Please enter 1, 2, or 3.")

def categorize_files(filenames):
    """Categorize files by type for better display."""
    wdl_files = [f for f in filenames if f.endswith('.rtbw')]
    dtz_files = [f for f in filenames if f.endswith('.rtbz')]
    return wdl_files, dtz_files

def main():
    """Main function to orchestrate the download process."""
    parser = argparse.ArgumentParser(description='Download Syzygy 3-4-5 tablebase files')
    parser.add_argument('--wdl-only', action='store_true', help='Download only WDL (.rtbw) files')
    parser.add_argument('--dtz-only', action='store_true', help='Download only DTZ (.rtbz) files')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')

    args = parser.parse_args()

    print("=== Syzygy 3-4-5 Tablebase Downloader ===")
    print()

    # Determine file types to download
    if args.wdl_only:
        file_types = ['rtbw']
    elif args.dtz_only:
        file_types = ['rtbz']
    else:
        if args.yes:
            file_types = ['rtbw', 'rtbz']  # Default to both if --yes is used
        else:
            file_types = get_file_type_choice()

    # Create target directory
    create_target_directory(TARGET_DIR)

    # Fetch and parse directory listing
    html_content = fetch_directory_listing(BASE_URL)
    filenames = extract_filenames(html_content, file_types)

    if not filenames:
        print("No tablebase files found in the directory listing!")
        sys.exit(1)

    # Categorize and display files
    wdl_files, dtz_files = categorize_files(filenames)

    print(f"Found tablebase files to download:")
    if 'rtbw' in file_types and wdl_files:
        print(f"  📊 WDL files (.rtbw): {len(wdl_files)}")
    if 'rtbz' in file_types and dtz_files:
        print(f"  🎯 DTZ files (.rtbz): {len(dtz_files)}")
    print(f"  📁 Total files: {len(filenames)}")
    print()

    # Show sample files
    print("Sample files:")
    for i, filename in enumerate(filenames[:10]):  # Show first 10
        file_type = "WDL" if filename.endswith('.rtbw') else "DTZ"
        print(f"  {i+1:3d}. {filename} ({file_type})")
    if len(filenames) > 10:
        print(f"  ... and {len(filenames) - 10} more")
    print()

    # Confirm with user
    if not args.yes:
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

    if 'rtbw' in file_types and 'rtbz' in file_types:
        wdl_downloaded = len([f for f in filenames if f.endswith('.rtbw') and
                             os.path.exists(os.path.join(TARGET_DIR, f))])
        dtz_downloaded = len([f for f in filenames if f.endswith('.rtbz') and
                             os.path.exists(os.path.join(TARGET_DIR, f))])
        print(f"📊 WDL files: {wdl_downloaded}")
        print(f"🎯 DTZ files: {dtz_downloaded}")

if __name__ == "__main__":
    main()
