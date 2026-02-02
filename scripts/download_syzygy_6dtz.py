#!/usr/bin/env python3
"""
Script to download all Syzygy 6-DTZ tablebase files from tablebase.sesse.net
Contains DTZ (.rtbz) files for 6-piece endgames
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
BASE_URL = "http://tablebase.sesse.net/syzygy/6-DTZ/"
TARGET_DIR = "syzygy_tables/6-DTZ"

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
    """Extract all .rtbz filenames from the Apache directory listing HTML."""
    # Pattern to match .rtbz files in Apache directory listing
    # Look for links with .rtbz extension
    pattern = r'<a href="([^"]+\.rtbz)">'
    filenames = re.findall(pattern, html_content, re.IGNORECASE)

    # Alternative pattern in case the first one doesn't work
    if not filenames:
        # Look for .rtbz files in a more general way
        pattern = r'([A-Z][A-Za-z0-9]+\.rtbz)'
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

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes}B"

def estimate_total_size(filenames):
    """Estimate total download size based on typical 6-piece DTZ file sizes."""
    # Rough estimates based on common 6-piece DTZ file sizes
    total_bytes = 0
    for filename in filenames:
        # Simple heuristic: files with more pieces tend to be larger
        piece_count = len([c for c in filename if c in 'BRNQP'])
        if piece_count <= 2:
            total_bytes += 50 * 1024 * 1024  # ~50MB
        elif piece_count <= 4:
            total_bytes += 200 * 1024 * 1024  # ~200MB
        else:
            total_bytes += 500 * 1024 * 1024  # ~500MB

    return total_bytes

def main():
    """Main function to orchestrate the download process."""
    parser = argparse.ArgumentParser(description='Download Syzygy 6-DTZ tablebase files')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without downloading')

    args = parser.parse_args()

    print("=== Syzygy 6-DTZ Tablebase Downloader ===")
    print("📊 DTZ (Distance to Zero) files for 6-piece endgames")
    print()

    # Create target directory
    create_target_directory(TARGET_DIR)

    # Fetch and parse directory listing
    html_content = fetch_directory_listing(BASE_URL)
    filenames = extract_filenames(html_content)

    if not filenames:
        print("No .rtbz files found in the directory listing!")
        sys.exit(1)

    # Show file information
    print(f"Found {len(filenames)} DTZ tablebase files to download")

    # Estimate total size
    estimated_size = estimate_total_size(filenames)
    print(f"🗄️  Estimated total size: ~{format_file_size(estimated_size)}")
    print()

    # Show sample files
    print("Sample files:")
    for i, filename in enumerate(filenames[:10]):  # Show first 10
        print(f"  {i+1:3d}. {filename}")
    if len(filenames) > 10:
        print(f"  ... and {len(filenames) - 10} more")
    print()

    if args.dry_run:
        print("🔍 Dry run mode - showing what would be downloaded:")
        for filename in filenames:
            print(f"  Would download: {filename}")
        print(f"\nTotal files: {len(filenames)}")
        print(f"Estimated size: ~{format_file_size(estimated_size)}")
        return

    # Confirm with user
    if not args.yes:
        print("⚠️  Note: 6-piece DTZ files are very large and the download may take several hours.")
        response = input(f"Download all {len(filenames)} files to {TARGET_DIR}? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Download cancelled.")
            sys.exit(0)

    # Download files
    print(f"\nStarting downloads...")
    print("=" * 50)

    successful = 0
    failed = 0
    total_downloaded = 0

    for i, filename in enumerate(filenames, 1):
        print(f"\n[{i}/{len(filenames)}] ", end="")

        if download_file(filename, BASE_URL, TARGET_DIR):
            successful += 1
            # Add to total downloaded size
            target_path = os.path.join(TARGET_DIR, filename)
            if os.path.exists(target_path):
                total_downloaded += os.path.getsize(target_path)
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
    print(f"💾 Total downloaded: {format_file_size(total_downloaded)}")

    if failed > 0:
        print(f"\n⚠️  {failed} files failed to download. You can re-run this script to retry failed downloads.")

if __name__ == "__main__":
    main()
