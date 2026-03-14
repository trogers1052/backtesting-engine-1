"""
Checkpoint support for resumable validation runs.

Saves completed symbol results to JSON so interrupted runs can resume.
"""
import json
import os
from datetime import date
from typing import List, Optional


class CheckpointManager:
    """Save/load validation progress for resumability."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir or os.path.expanduser(
            "~/.backtesting/checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_run_id(self) -> str:
        """Generate run ID from today's date."""
        return date.today().isoformat()

    def _run_dir(self, run_id: str) -> str:
        path = os.path.join(self.checkpoint_dir, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_symbol_complete(self, run_id: str, symbol: str, summary: dict):
        """Mark a symbol as completed with summary data."""
        filepath = os.path.join(self._run_dir(run_id), f"{symbol}.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def get_completed_symbols(self, run_id: str) -> List[str]:
        """Get list of symbols already completed in this run."""
        run_dir = self._run_dir(run_id)
        completed = []
        for fname in os.listdir(run_dir):
            if fname.endswith(".json"):
                completed.append(fname.replace(".json", ""))
        return completed

    def load_symbol_summary(self, run_id: str, symbol: str) -> Optional[dict]:
        """Load saved summary for a symbol."""
        filepath = os.path.join(self._run_dir(run_id), f"{symbol}.json")
        if os.path.exists(filepath):
            with open(filepath) as f:
                return json.load(f)
        return None

    def clear_run(self, run_id: str):
        """Delete all checkpoints for a run."""
        run_dir = self._run_dir(run_id)
        if os.path.exists(run_dir):
            for fname in os.listdir(run_dir):
                os.remove(os.path.join(run_dir, fname))
            os.rmdir(run_dir)
