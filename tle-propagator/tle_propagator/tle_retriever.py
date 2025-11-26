"""
TLE Retriever module to fetch TLE data from Celestrak.
"""
from pathlib import Path

import requests


def retrieve_tle(norad_id: str) -> Path:
    """Retrieve TLE data for a given NORAD ID from Celestrak.
    
    Args:
        norad_id (str): The NORAD ID of the satellite.
    Returns:
        Path: Path to the saved TLE file.
    """
    r = requests.get("https://celestrak.org/NORAD/elements/gp.php", params={"CATNR": norad_id}, timeout=20)
    input_dir = Path("input")
    input_dir.mkdir(parents=True, exist_ok=True)
    gp_file = input_dir / f"celestrak_{norad_id}.txt"
    r.raise_for_status()
    if r.text.strip() == "No GP data found":
        raise (f"No GP data found for NORAD ID {norad_id}") # type: ignore
    with open(gp_file, "w", encoding=r.encoding or "utf-8", newline="\n") as f:
        f.write(r.text)
    return gp_file