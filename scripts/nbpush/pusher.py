"""
Cloud pusher - handles pushing notebooks to cloud services.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .logger import get_logger
from .scanner import NotebookInfo


class CloudPusher:
    """Handles pushing notebooks to cloud services."""

    def __init__(self, project_root: Path = None):
        """Initialize pusher with project root."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)
        self.logger = get_logger()

    def push_to_kaggle(self, notebook: NotebookInfo, dry_run: bool = False) -> Dict[str, Any]:
        """
        Push notebook to Kaggle.

        Args:
            notebook: Notebook to push
            dry_run: If True, only show what would be done

        Returns:
            Result dict with status and output
        """
        self.logger.log(
            action="push",
            status="started",
            notebook=notebook.relative_path,
            service="kaggle",
            metadata={"dry_run": dry_run}
        )

        # Determine kernel directory
        kernel_dir = notebook.path.parent

        # Check for kernel-metadata.json
        metadata_file = kernel_dir / "kernel-metadata.json"
        if not metadata_file.exists():
            error = f"No kernel-metadata.json found in {kernel_dir}"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service="kaggle",
                error=error
            )
            return {"success": False, "error": error}

        if dry_run:
            result = {
                "success": True,
                "dry_run": True,
                "kernel_dir": str(kernel_dir),
                "command": f"cd {kernel_dir} && KAGGLE_CONFIG_DIR=../../../.kaggle kaggle kernels push"
            }
            self.logger.log(
                action="push",
                status="completed",
                notebook=notebook.relative_path,
                service="kaggle",
                metadata=result
            )
            return result

        # Execute kaggle push
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "push"],
                cwd=kernel_dir,
                env={**subprocess.os.environ, "KAGGLE_CONFIG_DIR": str(self.project_root / ".kaggle")},
                capture_output=True,
                text=True,
                timeout=30
            )

            success = result.returncode == 0
            output = {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "kernel_dir": str(kernel_dir)
            }

            self.logger.log(
                action="push",
                status="completed" if success else "failed",
                notebook=notebook.relative_path,
                service="kaggle",
                metadata=output,
                error=result.stderr if not success else None
            )

            return output

        except subprocess.TimeoutExpired:
            error = "Kaggle push timed out after 30 seconds"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service="kaggle",
                error=error
            )
            return {"success": False, "error": error}

        except Exception as e:
            error = f"Kaggle push failed: {str(e)}"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service="kaggle",
                error=error
            )
            return {"success": False, "error": error}

    def push_to_colab(self, notebook: NotebookInfo, dry_run: bool = False) -> Dict[str, Any]:
        """
        Push notebook to Colab.

        Args:
            notebook: Notebook to push
            dry_run: If True, only show what would be done

        Returns:
            Result dict with status and output
        """
        self.logger.log(
            action="push",
            status="started",
            notebook=notebook.relative_path,
            service="colab",
            metadata={"dry_run": dry_run}
        )

        if dry_run:
            result = {
                "success": True,
                "dry_run": True,
                "command": f"colab-cli push-nb {notebook.path}"
            }
            self.logger.log(
                action="push",
                status="completed",
                notebook=notebook.relative_path,
                service="colab",
                metadata=result
            )
            return result

        # Execute colab push
        try:
            result = subprocess.run(
                ["colab-cli", "push-nb", str(notebook.path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            success = result.returncode == 0
            output = {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            self.logger.log(
                action="push",
                status="completed" if success else "failed",
                notebook=notebook.relative_path,
                service="colab",
                metadata=output,
                error=result.stderr if not success else None
            )

            return output

        except subprocess.TimeoutExpired:
            error = "Colab push timed out after 60 seconds"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service="colab",
                error=error
            )
            return {"success": False, "error": error}

        except Exception as e:
            error = f"Colab push failed: {str(e)}"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service="colab",
                error=error
            )
            return {"success": False, "error": error}

    def push(self, notebook: NotebookInfo, service: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Push notebook to specified cloud service.

        Args:
            notebook: Notebook to push
            service: Cloud service ('kaggle' or 'colab')
            dry_run: If True, only show what would be done

        Returns:
            Result dict with status and output
        """
        if service == "kaggle":
            return self.push_to_kaggle(notebook, dry_run)
        elif service == "colab":
            return self.push_to_colab(notebook, dry_run)
        else:
            error = f"Unknown service: {service}"
            self.logger.log(
                action="push",
                status="failed",
                notebook=notebook.relative_path,
                service=service,
                error=error
            )
            return {"success": False, "error": error}
