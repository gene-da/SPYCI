#!/usr/bin/env python3
"""
lcsc_getter_launcher.py
Main entry for packaging as an .exe (PyInstaller-friendly).

- On first run, asks where to store your KiCad library and saves it in a config file.
- On subsequent runs, uses the saved path (with an option to change it via "config").
- Runs `easyeda2kicad --full --lcsc_id=<ID> --output <library_path>` for each ID you enter.

Pro tip: after you build the .exe, throw it on your PATH and stop babysitting virtualenvs.
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

APP_NAME = "lcsc_getter"
DEFAULT_LIBRARY = str(Path.home() / "Documents" / "dev" / "KiCad_Libs" / "LCSC")

def config_dir() -> Path:
    # Windows-first. If APPDATA is missing (you wizard), fall back to home.
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / APP_NAME

def config_path() -> Path:
    return config_dir() / "config.json"

def load_config() -> dict:
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Config read failed ({e}). Using defaults.")
        return {}

def save_config(cfg: dict) -> None:
    cfg_dir = config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path(), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def prompt_library_path(current: str | None = None) -> str:
    print("\n=== Library Output Folder ===")
    if current:
        print(f"Current library path: {current}")
        resp = input("Keep this? [Y/n]: ").strip().lower()
        if resp in ("", "y", "yes"):
            return current
    # Ask for a new path
    while True:
        proposed = input(f"Enter library folder path [{DEFAULT_LIBRARY}]: ").strip()
        if not proposed:
            proposed = DEFAULT_LIBRARY
        p = Path(proposed).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            print(f"Using library folder: {p}")
            return str(p)
        except Exception as e:
            print(f"Couldn't create/use that path: {e}")
            print("Try again.")

def ensure_easyeda2kicad_available() -> str | None:
    # If the tool is on PATH, great. Otherwise, tell the user what to do.
    exe = "easyeda2kicad.exe" if os.name == "nt" else "easyeda2kicad"
    resolved = shutil.which(exe)
    if resolved:
        return resolved
    print("\nHeads up: 'easyeda2kicad' was not found on your PATH.")
    print("Install it and make sure the command is available. Example:")
    print("  pip install easyeda2kicad")
    print("Then re-run this program.")
    return None

def run_conversion(tool: str, lcsc_id: str, out_dir: str) -> int:
    cmd = [
        tool,
        "--full",
        f"--lcsc_id={lcsc_id}",
        "--output",
        out_dir,
    ]
    print(f"\n> Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)  # stream output directly
        return result.returncode
    except FileNotFoundError:
        print("Failed to launch easyeda2kicad (not found).")
        return 127
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

def handle_config(cfg: dict) -> dict:
    # Offer quick config mode if user launched with "config"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "config":
        cfg["library_path"] = prompt_library_path(cfg.get("library_path"))
        save_config(cfg)
        print("Config updated.")
        # If they only wanted to set config, exit early
        if len(sys.argv) == 2:
            sys.exit(0)
    return cfg

def main():
    cfg = load_config()
    cfg = handle_config(cfg)

    # Ensure library path is set
    if not cfg.get("library_path"):
        cfg["library_path"] = prompt_library_path(None)
        save_config(cfg)
    else:
        # Confirm or change
        cfg["library_path"] = prompt_library_path(cfg["library_path"])
        save_config(cfg)

    tool = ensure_easyeda2kicad_available()
    if not tool:
        # Tool missing; no point continuing
        sys.exit(127)

    print("\nType an LCSC ID (e.g., C12345). Commands: 'config' to change paths, 'quit' to exit.")
    while True:
        lcsc_id = input("\nLCSC ID> ").strip()

        if not lcsc_id:
            print("Please enter a valid LCSC ID (or 'quit').")
            continue

        low = lcsc_id.lower()
        if low in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if low == "config":
            cfg["library_path"] = prompt_library_path(cfg.get("library_path"))
            save_config(cfg)
            continue

        # Minimal validation: allow C-prefix or just digits
        if not (lcsc_id.upper().startswith("C") or lcsc_id.isdigit()):
            print("That doesn't look like an LCSC ID. Example: C12345")
            continue

        code = run_conversion(tool, lcsc_id, cfg["library_path"])
        if code == 0:
            print("✅ Done.")
        else:
            print(f"⚠️ Command exited with code {code}.")

if __name__ == "__main__":
    main()
