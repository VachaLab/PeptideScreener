import torch
from typing import List, Tuple
from pathlib import Path
import re
import shutil

def get_best_device(options:List):
    """
    Returns the best available device in this order of preference:
      1. CUDA (NVIDIA GPU) if available
      2. MPS (Apple Silicon GPU) if available
      3. CPU (fallback)
    """
    if torch.cuda.is_available() and 'cuda' in options:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        
    elif torch.backends.mps.is_available() and 'mps' in options:
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
        
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device

def get_next_run_id(
    base_dir: Path,
    pattern: str = r".*?(\d+).*",
    delete_last: bool = True,
    thr: int = 3
) -> Tuple[int, int]:
    """
    Finds the highest existing run_XXXX number in base_dir and returns the next number.
    If delete_last=True and number of existing dirs > thr, deletes the directory
    with the lowest number before returning the next ID.

    Returns:
        Tuple[next_id: int, count_after_operation: int]
        
    Example:
        Existing: run_003, run_007, run_042
        → returns (43, 3)
        
        With delete_last=True, thr=3 and 4 directories exist:
        Deletes the lowest one (e.g. run_003), then returns next id + new count (3)
    """
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return 1, 0

    existing = []
    
    # Collect all matching directories and their numbers
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        match = re.match(pattern, item.name)
        if match:
            try:
                num = int(match.group(1))
                existing.append((num, item))
            except ValueError:
                continue

    if not existing:
        return 1, 0

    # Sort by number (smallest to largest)
    existing.sort(key=lambda x: x[0])
    current_count = len(existing)

    # Delete oldest (lowest number) if needed
    if delete_last and current_count > thr:
        oldest_num, oldest_path = existing[0]
        try:
            shutil.rmtree(oldest_path)
            print(f"Deleted oldest run directory: {oldest_path.name} (#{oldest_num})")
            current_count -= 1
        except Exception as e:
            print(f"Warning: Failed to delete {oldest_path.name}: {e}")

    next_id = max(num for num, _ in existing) + 1
    return next_id, current_count