"""
Notebook scanner - finds and lists notebooks in project.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class NotebookInfo:
    """Information about a notebook."""
    path: Path
    name: str
    relative_path: str
    modified: datetime
    size_mb: float
    platform: str  # 'local', 'kaggle', 'colab'

    def __str__(self) -> str:
        """String representation."""
        age = datetime.now() - self.modified
        if age.days > 0:
            age_str = f"{age.days}d ago"
        elif age.seconds > 3600:
            age_str = f"{age.seconds // 3600}h ago"
        elif age.seconds > 60:
            age_str = f"{age.seconds // 60}m ago"
        else:
            age_str = "just now"

        return (
            f"{self.relative_path:<50} "
            f"[{self.platform:^6}] "
            f"{self.size_mb:>5.1f}MB "
            f"{age_str:>10}"
        )


class NotebookScanner:
    """Scans project directories for notebooks."""

    def __init__(self, project_root: Path = None):
        """Initialize scanner with project root."""
        if project_root is None:
            # Default to project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)

        # Define scan directories and their platforms
        self.scan_dirs = {
            "jupyter_notebooks": "local",
            "kaggle/kernels": "kaggle",
            "colab": "colab",
        }

    def scan(self) -> List[NotebookInfo]:
        """
        Scan for notebooks in all configured directories.

        Returns:
            List of NotebookInfo objects, sorted by modification time (newest first)
        """
        notebooks = []

        for scan_dir, platform in self.scan_dirs.items():
            dir_path = self.project_root / scan_dir

            if not dir_path.exists():
                continue

            # Find all .ipynb files recursively
            for notebook_path in dir_path.rglob("*.ipynb"):
                # Skip checkpoint files
                if ".ipynb_checkpoints" in str(notebook_path):
                    continue

                # Get file stats
                stat = notebook_path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime)
                size_mb = stat.st_size / (1024 * 1024)

                # Create relative path from project root
                try:
                    relative_path = notebook_path.relative_to(self.project_root)
                except ValueError:
                    relative_path = notebook_path

                notebooks.append(
                    NotebookInfo(
                        path=notebook_path,
                        name=notebook_path.name,
                        relative_path=str(relative_path),
                        modified=modified,
                        size_mb=size_mb,
                        platform=platform,
                    )
                )

        # Sort by modification time (newest first)
        notebooks.sort(key=lambda n: n.modified, reverse=True)

        return notebooks

    def find_notebook(self, search: str) -> NotebookInfo | None:
        """
        Find a notebook by name or path.

        Args:
            search: Notebook name or relative path

        Returns:
            NotebookInfo if found, None otherwise
        """
        notebooks = self.scan()

        # Try exact name match first
        for nb in notebooks:
            if nb.name == search or nb.name == f"{search}.ipynb":
                return nb

        # Try relative path match
        for nb in notebooks:
            if nb.relative_path == search or nb.relative_path.endswith(search):
                return nb

        # Try partial match
        search_lower = search.lower()
        for nb in notebooks:
            if search_lower in nb.name.lower() or search_lower in nb.relative_path.lower():
                return nb

        return None

    def get_by_platform(self, platform: str) -> List[NotebookInfo]:
        """
        Get notebooks filtered by platform.

        Args:
            platform: Platform name ('local', 'kaggle', 'colab')

        Returns:
            List of notebooks for that platform
        """
        notebooks = self.scan()
        return [nb for nb in notebooks if nb.platform == platform]
