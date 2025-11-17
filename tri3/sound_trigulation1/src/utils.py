import os
import time
from typing import Optional


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    for remaining in range(seconds, 0, -1):
        print(f"Recording starts in {remaining}...")
        time.sleep(1)
    print("Recording now!")


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


