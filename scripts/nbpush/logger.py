"""
Activity logger optimized for LLM consumption.

Logs all CLI activity in JSON Lines format (.jsonl) for easy parsing by AI assistants.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ActivityLogger:
    """Log all CLI activity in LLM-friendly format."""

    def __init__(self, log_dir: Path = None):
        """Initialize logger with log directory."""
        if log_dir is None:
            # Default to project root .logs directory
            log_dir = Path(__file__).parent.parent.parent / ".logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "nbpush_activity.jsonl"

    def log(
        self,
        action: str,
        status: str,
        notebook: Optional[str] = None,
        service: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log an activity event in JSON Lines format.

        Args:
            action: Action performed (list, push, status, download)
            status: Status (started, completed, failed, info)
            notebook: Notebook path/name
            service: Cloud service (kaggle, colab)
            metadata: Additional metadata dict
            error: Error message if failed
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timestamp_local": datetime.now().isoformat(),
            "action": action,
            "status": status,
        }

        if notebook:
            entry["notebook"] = notebook

        if service:
            entry["service"] = service

        if metadata:
            entry["metadata"] = metadata

        if error:
            entry["error"] = error

        # Write as single-line JSON
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, separators=(',', ':')) + "\n")

    def get_recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        """
        Get recent activity entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of log entries (most recent first)
        """
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Return most recent first
        return list(reversed(entries))[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of recent activity.

        Returns:
            Summary dict with counts and recent actions
        """
        entries = self.get_recent(limit=100)

        if not entries:
            return {"total_actions": 0, "recent_actions": []}

        # Count by action and status
        action_counts = {}
        status_counts = {}
        service_counts = {}

        for entry in entries:
            action = entry.get("action", "unknown")
            status = entry.get("status", "unknown")
            service = entry.get("service")

            action_counts[action] = action_counts.get(action, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1

            if service:
                service_counts[service] = service_counts.get(service, 0) + 1

        return {
            "total_actions": len(entries),
            "action_counts": action_counts,
            "status_counts": status_counts,
            "service_counts": service_counts,
            "recent_actions": entries[:10],  # Last 10 actions
            "last_activity": entries[0]["timestamp_local"] if entries else None,
        }


# Global logger instance
_logger = None

def get_logger() -> ActivityLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = ActivityLogger()
    return _logger
