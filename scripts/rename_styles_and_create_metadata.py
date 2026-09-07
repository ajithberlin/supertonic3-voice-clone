#!/usr/bin/env python3
"""Rename all cloned style JSON files in voices/cloned_styles to <voice_id>.json
and create comprehensive metadata files with:
voice_id, gender, name, filnename, status, speed, voice_characters, sampling_rate
"""

import json
import os
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
MANIFEST_PATH = WORKSPACE / "voices/voice/voice_samples.json"
STYLES_DIR = WORKSPACE / "voices/cloned_styles"

# Character speeds for slow/expressive pacing
CHARACTER_SPEEDS = {
    "CwhRBWXzGAHq8TQ4Fs17": 0.86,  # Roger
    "EXAVITQu4vr4xnSDxMaL": 0.87,  # Sarah
    "FGY2WhTYpPnrIDTdsKH5": 0.88,  # Laura
    "IKne3meq5aSn9XLyUdCD": 0.87,  # Charlie
    "JBFqnCBsd6RMkjVDRZzb": 0.85,  # George
    "N2lVS1w4EtoT3dr4eOWO": 0.86,  # Callum
    "SAz9YHcvj6GT2YYXdXww": 0.86,  # River
    "SOYHLrjzK2X1ezoPC6cr": 0.87,  # Harry
    "TX3LPaxmHKxFdv7VOQHJ": 0.89,  # Liam
    "Xb7hH8MSUJpSbSDYk0k2": 0.87,  # Alice
    "XrExE9yKIg1WjnnlVkGX": 0.86,  # Matilda
    "bIHbv24MWmeRgasZH58o": 0.86,  # Will
    "cgSgspJ2msm6clMCkdW9": 0.88,  # Jessica
    "cjVigY5qzO86Huf0OWal": 0.86,  # Eric
    "hpp4J3VqNfWAUOO0d1Us": 0.88,  # Bella
    "iP95p4xoKVk53GoZ742B": 0.87,  # Chris
    "nPczCjzI2devNBz1zQrb": 0.84,  # Brian
    "onwK4e9ZLuTAKqWW03F9": 0.88,  # Daniel
    "pFZP5JQG7iQjIQuC4Bku": 0.85,  # Lily
    "pNInz6obpgDQGcFmaJgB": 0.86,  # Adam
    "pqHfZKP75CvOlQylNhV4": 0.85,  # Bill
    "Uyx98Ek4uMNmWN7E28CD": 0.87,  # Aakash Aryan
    "auq43ws1oslv0tO4BDa7": 0.85,  # Adam Stone
    "GKDaBI8TKSBJVhsCLD6n": 0.86,  # Asahi
    "gPPH6SLdL8XSX6GNJ40G": 0.87,  # Brian
    "nwj0s2LU9bDWRKND5yzA": 0.89,  # Bunty
    "b98xdQeG04BqhjleDbOS": 0.85,  # Avani
    "L4bD71zGAYHMT7a6MLwc": 0.86,  # Brenna
    "LRpNiUBlcqgIsKUzcrlN": 0.88,  # Georg
    "sFEOLSczVqVpvB7COD8U": 0.89,  # Choy
    "EkK6wL8GaH8IgBZTTDGJ": 0.88,  # Akari
    "7qBNUtXRGP0jPi0H4r8k": 0.89,  # Bunty
    "0a3rU6OS52qFMvnAmGct": 0.86,  # Cory
    "UEKYgullGqaF0keqT8Bu": 0.87,  # Chris Brift
    "ngvNHfiCrXLPAHcTrZK1": 0.87,  # Aki
    "uYXf8XasLslADfZ2MB4u": 0.88,  # Hope
    "c2XJrw7TvNGtOc6r0ijG": 0.86,  # Harune
    "Bj4Malc5SZLoXfPtxRxH": 0.84,  # Hiro
    "r1KmysJdVYZjJCm4mL3b": 0.88,  # Jessica
    "JOcmGzB8OFjY8MhjHHEf": 0.85,  # Jun
    "urE3OJfJRxJuk9kAMN0Y": 0.86,  # Gojo
    "AAxVQbetQhXWMEZC9p8S": 0.87,  # Kmy
    "SSfU0eLfP3qeuR4j2bwD": 0.85,  # Christopher
    "d8K4L6ChE4wBDRh6uxtN": 0.86,  # Kou Sudo
    "H8ZPDxbrPcks5hEsi2fq": 0.84,  # Koichi
    "qbfQuoRJv1T3ei3qV4bc": 0.86,  # Kazuha
    "TcrlBgVmqvmPFJi2o2AO": 0.88,  # Kylie
    "RJNtO4oCanOuvQa3XSb7": 0.89,  # Ikuya
    "ZebtvV8hAiDrAt9nw3Z3": 0.87,  # Mira
    "CNs61ARiqwaYAbqKRbHf": 0.87,  # Ken
    "kdmDKE6EkgrWrrykO9Qt": 0.86,  # Alexandra
    "b34JylakFZPlGS0BnwyY": 0.85,  # Kenzo
    "8EkOjt4xTPGMclNlh1pk": 0.84,  # Morioki
    "Dgs8mKlgA2h114MjpKuS": 0.85,  # Priest
    "aRRpAlz9zwsfcqvwusHA": 0.86,  # Rie
    "zzahs6DT06h2pEM743ds": 0.88,  # Rie
    "eqMFh4kWVrmjm0Bcil6E": 0.88,  # Nguyen
    "pUgmTF2V1ptIKsYb6qON": 0.85,  # Ryo
    "4oeIbMTUt5QeJy4ZX1FC": 0.84,  # Koichi Yashiro
    "Raa94hHxcH2itBN60mKp": 0.87,  # Mei
    "qmzV4HKbcUWCnzV4CaHW": 0.87,  # Katelyn
    "EbuvaInXUGWtpYRUnKLQ": 0.85,  # Sawaro
    "8PfKHL4nZToWC3pbz9U9": 0.86,  # Rose
    "dtSEyYGNJqjrtBArPCVZ": 0.88,  # Titan
    "8FuuqoKHuM48hIEwni5e": 0.85,  # Shohei
    "iWNf11sz1GrUE4ppxTOL": 0.88,  # Viraj
    "5FNeYl6NmyAXYQWW7CEV": 0.86,  # Yoshiki
    "UwmChKysRWw5hwbzf6t6": 0.85,  # Suisen
    "ljX1ZrXuDIIRVcmiVSyR": 0.87,  # Michael
    "J6YFreR6shJoaDfv7tLf": 0.86,  # Yuko
    "g8HmbLedl5nLG1746E5V": 0.87,  # Yoshi M
    "80H5qqdq3bIO8OjFE6ue": 0.86,  # Yukari
    "JTlYtJrcTzPC71hMLOxo": 0.85,  # Yuki
    "k0oRjDwazplPRQKtXvUn": 0.85,  # Zach
    "0ptCJp0xgdabdcpVtCB5": 0.86,  # Yoko Honda
    "hkfHEbBvdQFNX4uWHqRF": 0.88,  # Stacy
    "ozfS3gQtjFX3kQyJ12dX": 0.86,  # Saori
    "HQ1o7gECNyaEC0RiqY4w": 0.84,  # Sui
}


def main():
    print("=== Renaming cloned style JSONs and generating metadata ===")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest["entries"]
    entry_by_voice_id = {e["voice_id"]: e for e in entries}

    # Find all style JSON files
    style_files = sorted(list(STYLES_DIR.glob("*.json")))
    print(f"Found {len(style_files)} style files in {STYLES_DIR}")

    renamed_count = 0
    already_named_count = 0
    metadata_list = []

    # Map each file to target <voice_id>.json
    file_to_target = {}
    for f in style_files:
        if f.name == "metadata.json" or f.name == "METADATA.json":
            continue
        stem = f.stem
        if stem in entry_by_voice_id:
            # Already named <voice_id>.json
            target_name = f.name
            entry = entry_by_voice_id[stem]
            already_named_count += 1
        else:
            idx = int(stem.split("_")[0])
            entry = entries[idx - 1]
            target_name = f"{entry['voice_id']}.json"
            renamed_count += 1
        file_to_target[f] = (target_name, entry)

    # Perform renames
    for src_path, (target_name, entry) in file_to_target.items():
        dst_path = STYLES_DIR / target_name
        if src_path != dst_path:
            src_path.rename(dst_path)

    print(f"Renamed {renamed_count} files to <voice_id>.json")
    print(f"Kept {already_named_count} files that already had <voice_id>.json")

    # Build comprehensive metadata ordered by original index
    for idx, entry in enumerate(entries, 1):
        voice_id = entry["voice_id"]
        filename = f"{voice_id}.json"
        
        # Check that the style file actually exists
        style_path = STYLES_DIR / filename
        file_exists = style_path.is_file()

        item = {
            "voice_id": voice_id,
            "gender": entry.get("gender") or "Not specified",
            "name": entry.get("name") or f"Voice_{idx}",
            "filnename": filename,
            "filename": filename,
            "status": "ok" if file_exists else "missing",
            "speed": CHARACTER_SPEEDS.get(voice_id, 0.86),
            "voice_characters": entry.get("voice_characters", ""),
            "sampling_rate": 44100,
        }
        metadata_list.append(item)

    # Save to multiple convenient locations:
    # 1. voices/metadata.json (root of voices directory)
    # 2. voices/cloned_styles_metadata.json (convenient root access)
    # 3. voices/cloned_styles/metadata.json (inside cloned_styles)
    out_paths = [
        WORKSPACE / "voices/metadata.json",
        WORKSPACE / "voices/cloned_styles_metadata.json",
        STYLES_DIR / "metadata.json",
    ]

    for p in out_paths:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to: {p}")

    print("\nVerification:")
    current_styles = sorted([f.name for f in STYLES_DIR.glob("*.json") if f.name != "metadata.json"])
    print(f"Total style files remaining: {len(current_styles)}")
    all_voice_ids = [e["voice_id"] + ".json" for e in entries]
    matches = set(current_styles) == set(all_voice_ids)
    print(f"All 78 styles exactly match <voice_id>.json: {matches}")
    print("=== Done ===")

if __name__ == "__main__":
    main()
