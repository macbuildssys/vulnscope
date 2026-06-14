"""
keystore.py
Securely store and retrieve the NVD API key in ~/.config/vulnscope/config.json.

The key is never written to the project directory, so it cannot accidentally
end up in git. File permissions are set to 0600 (user-read/write only).
"""

import getpass
import json
import os
import stat
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".config" / "vulnscope"
CONFIG_FILE = CONFIG_DIR / "config.json"

NVD_KEY_FIELD = "nvd_api_key"

def _read_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

def _write_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)

def get_api_key() -> Optional[str]:
    # Return the stored NVD API key, or None if not set
    return _read_config().get(NVD_KEY_FIELD)

def save_api_key(key: str) -> None:
    """Persist *key* to the config file."""
    cfg = _read_config()
    cfg[NVD_KEY_FIELD] = key.strip()
    _write_config(cfg)

def delete_api_key() -> None:
    # Remove the stored API key
    cfg = _read_config()
    cfg.pop(NVD_KEY_FIELD, None)
    _write_config(cfg)

def prompt_for_key(allow_skip: bool = False) -> Optional[str]:
    """
    Interactive terminal prompt for the NVD API key.
    Uses getpass so the key is never echoed to the terminal.
    Returns the entered key, or None if the user chose to skip.
    """
    _print_banner()
    existing = get_api_key()
    if existing:
        masked = existing[:6] + "..." + existing[-4:]
        print(f"  Stored key detected: {masked}")
        choice = input("  Use existing key? [Y/n] ").strip().lower()
        if choice in ("", "y", "yes"):
            return existing
        print()

    print("  Get a free key at: https://nvd.nist.gov/developers/request-an-api-key")
    print("  (Key raises rate limit from 5 to 50 requests / 30 s)\n")

    while True:
        key = getpass.getpass("  Paste your NVD API key (input hidden): ").strip()
        if not key:
            if allow_skip:
                print("  Skipping API key. Rate-limited mode (slower download).\n")
                return None
            print("  Key cannot be empty. Try again, or press Ctrl-C to abort.")
            continue
        if len(key) < 20:
            print("  That looks too short for an NVD API key. Try again.")
            continue
        break

    save_choice = input("  Save key for future runs? [Y/n] ").strip().lower()
    if save_choice in ("", "y", "yes"):
        save_api_key(key)
        print(f"  Key saved to {CONFIG_FILE} (permissions: 600)\n")
    else:
        print("  Key not saved. You will be prompted again next run.\n")

    return key

def _print_banner():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║      VulnScope. NVD API Setup.        ║")
    print("  ╚══════════════════════════════════════╝")
    print()
