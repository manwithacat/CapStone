"""
CLI for pushing notebooks to cloud GPU services.

Usage:
    nbpush list                          # List all notebooks
    nbpush list --platform kaggle        # List Kaggle notebooks only
    nbpush push <notebook> --service kaggle   # Push to Kaggle
    nbpush push <notebook> --service colab    # Push to Colab
    nbpush activity                      # Show recent activity
    nbpush activity --summary            # Show activity summary
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import typer
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .scanner import NotebookScanner
from .pusher import CloudPusher
from .logger import get_logger


def extract_notebook_config(notebook_path: Path) -> Dict[str, Any]:
    """
    Extract CONFIG values from a notebook.

    Returns dict with:
        - use_sample: bool or None
        - sample_size: int or None
        - batch_size: int or None
        - epochs_stage1: int or None
        - epochs_stage2: int or None
    """
    config = {}

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_json = json.load(f)

        # Search through all code cells
        for cell in notebook_json.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Look for CONFIG = { ... } dictionary
                if 'CONFIG' in source and ('use_sample' in source or 'sample_size' in source):
                    # Extract use_sample
                    match = re.search(r"['\"]use_sample['\"]\s*:\s*(True|False)", source)
                    if match:
                        config['use_sample'] = match.group(1) == 'True'

                    # Extract sample_size
                    match = re.search(r"['\"]sample_size['\"]\s*:\s*(\d+)", source)
                    if match:
                        config['sample_size'] = int(match.group(1))

                    # Extract batch_size
                    match = re.search(r"['\"]batch_size['\"]\s*:\s*(\d+)", source)
                    if match:
                        config['batch_size'] = int(match.group(1))

                    # Extract epochs
                    match = re.search(r"['\"]epochs_stage1['\"]\s*:\s*(\d+)", source)
                    if match:
                        config['epochs_stage1'] = int(match.group(1))

                    match = re.search(r"['\"]epochs_stage2['\"]\s*:\s*(\d+)", source)
                    if match:
                        config['epochs_stage2'] = int(match.group(1))

                    # If we found config values, we're done
                    if config:
                        break

    except Exception:
        # If anything goes wrong, just return empty config
        pass

    return config

app = typer.Typer(
    name="nbpush",
    help="Push notebooks to cloud GPU services (Kaggle/Colab)",
    add_completion=False
)

console = Console()


@app.command()
def list(
    platform: Optional[str] = typer.Option(None, "--platform", "-p", help="Filter by platform (local/kaggle/colab)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of notebooks to show")
):
    """List available notebooks sorted by most recent modification."""
    logger = get_logger()
    logger.log(
        action="list",
        status="started",
        metadata={"platform": platform, "limit": limit}
    )

    scanner = NotebookScanner()

    if platform:
        notebooks = scanner.get_by_platform(platform)
    else:
        notebooks = scanner.scan()

    notebooks = notebooks[:limit]

    if not notebooks:
        console.print("[yellow]No notebooks found[/yellow]")
        logger.log(action="list", status="completed", metadata={"count": 0})
        return

    # Create rich table
    table = Table(
        title=f"Notebooks ({len(notebooks)} found)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Notebook", style="bold")
    table.add_column("Platform", justify="center", style="magenta")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Modified", justify="right", style="blue")

    for idx, nb in enumerate(notebooks, 1):
        # Calculate age
        from datetime import datetime
        age = datetime.now() - nb.modified
        if age.days > 0:
            age_str = f"{age.days}d ago"
        elif age.seconds > 3600:
            age_str = f"{age.seconds // 3600}h ago"
        elif age.seconds > 60:
            age_str = f"{age.seconds // 60}m ago"
        else:
            age_str = "just now"

        table.add_row(
            str(idx),
            nb.relative_path,
            nb.platform,
            f"{nb.size_mb:.1f}MB",
            age_str
        )

    console.print(table)

    logger.log(
        action="list",
        status="completed",
        metadata={
            "count": len(notebooks),
            "platform": platform,
            "notebooks": [nb.relative_path for nb in notebooks[:5]]  # Log first 5
        }
    )


@app.command()
def push(
    notebook: str = typer.Argument(...),
    service: str = typer.Option(..., "--service", "-s", help="Cloud service (kaggle/colab)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """
    Push a notebook to a cloud GPU service.

    NOTEBOOK: Notebook name or path to push
    """
    logger = get_logger()

    # Find notebook
    scanner = NotebookScanner()
    nb = scanner.find_notebook(notebook)

    if not nb:
        console.print(f"[red]❌ Notebook not found: {notebook}[/red]")
        logger.log(
            action="push",
            status="failed",
            notebook=notebook,
            service=service,
            error="Notebook not found"
        )
        raise typer.Exit(1)

    # Show what we're pushing
    panel = Panel(
        f"[bold]{nb.relative_path}[/bold]\n"
        f"Platform: [magenta]{nb.platform}[/magenta]\n"
        f"Size: [green]{nb.size_mb:.1f}MB[/green]\n"
        f"Target: [cyan]{service}[/cyan]\n"
        f"Dry run: [yellow]{dry_run}[/yellow]",
        title="📤 Push Notebook",
        border_style="cyan"
    )
    console.print(panel)

    # Push
    pusher = CloudPusher()
    result = pusher.push(nb, service, dry_run=dry_run)

    if result.get("success"):
        if dry_run:
            console.print("\n[yellow]🔍 Dry run - would execute:[/yellow]")
            console.print(f"[dim]{result.get('command')}[/dim]")
        else:
            console.print("\n[green]✅ Push successful![/green]")

            if result.get("stdout"):
                console.print("\n[dim]Output:[/dim]")
                console.print(result["stdout"])
    else:
        console.print(f"\n[red]❌ Push failed: {result.get('error')}[/red]")

        if result.get("stderr"):
            console.print("\n[dim]Error output:[/dim]")
            console.print(result["stderr"])

        raise typer.Exit(1)


@app.command()
def activity(
    summary: bool = typer.Option(False, "--summary", help="Show summary instead of detailed log"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent entries to show")
):
    """Show recent activity log."""
    logger = get_logger()

    if summary:
        # Show summary statistics
        stats = logger.get_summary()

        console.print(Panel(
            f"[bold]Total Actions:[/bold] {stats['total_actions']}\n"
            f"[bold]Last Activity:[/bold] {stats.get('last_activity', 'None')}\n\n"
            f"[bold cyan]Actions:[/bold cyan]\n" +
            "\n".join(f"  {action}: {count}" for action, count in stats['action_counts'].items()) +
            f"\n\n[bold cyan]Status:[/bold cyan]\n" +
            "\n".join(f"  {status}: {count}" for status, count in stats['status_counts'].items()) +
            (f"\n\n[bold cyan]Services:[/bold cyan]\n" +
             "\n".join(f"  {service}: {count}" for service, count in stats['service_counts'].items())
             if stats['service_counts'] else ""),
            title="📊 Activity Summary",
            border_style="green"
        ))

        # Show recent actions
        console.print("\n[bold cyan]Recent Actions:[/bold cyan]")
        for entry in stats['recent_actions']:
            status_color = {
                "completed": "green",
                "started": "blue",
                "failed": "red",
                "info": "yellow"
            }.get(entry.get("status", ""), "white")

            console.print(
                f"  [{status_color}]●[/{status_color}] "
                f"{entry['timestamp_local'][:19]} - "
                f"[bold]{entry['action']}[/bold] "
                f"({entry.get('notebook', 'N/A')[:50]}) "
                f"[{status_color}]{entry['status']}[/{status_color}]"
            )
    else:
        # Show detailed log
        entries = logger.get_recent(limit=limit)

        if not entries:
            console.print("[yellow]No activity logged yet[/yellow]")
            return

        table = Table(
            title=f"Recent Activity ({len(entries)} entries)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Time", style="dim")
        table.add_column("Action", style="bold")
        table.add_column("Notebook", style="blue")
        table.add_column("Service", style="magenta")
        table.add_column("Status")

        for entry in entries:
            status = entry.get("status", "")
            status_color = {
                "completed": "green",
                "started": "blue",
                "failed": "red",
                "info": "yellow"
            }.get(status, "white")

            notebook = entry.get("notebook", "-")
            if len(notebook) > 40:
                notebook = "..." + notebook[-37:]

            table.add_row(
                entry["timestamp_local"][:19],
                entry["action"],
                notebook,
                entry.get("service", "-"),
                f"[{status_color}]{status}[/{status_color}]"
            )

        console.print(table)

    logger.log(action="activity", status="info", metadata={"summary": summary, "limit": limit})


@app.command()
def select(
    service: Optional[str] = typer.Option(None, "--service", "-s", help="Cloud service (kaggle/colab) - skip service selection"),
    notebook: Optional[str] = typer.Option(None, "--notebook", "-n", help="Notebook name - skip notebook selection"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """
    Interactive notebook push with service and notebook selection.

    Use CLI options to bypass interactive prompts.
    """
    logger = get_logger()
    scanner = NotebookScanner()

    # Track original CLI parameters (before any modification)
    user_provided_service = service is not None
    user_provided_notebook = notebook is not None

    # Step 1: Select service (if not provided)
    if service is None:
        service = questionary.select(
            "Select cloud service:",
            choices=[
                questionary.Choice("Kaggle (T4 GPU, 9hr limit)", value="kaggle"),
                questionary.Choice("Colab (A100 GPU, 24hr limit)", value="colab"),
            ],
            style=questionary.Style([
                ('selected', 'bold cyan'),
                ('pointer', 'bold cyan'),
            ])
        ).ask()

        if service is None:  # User cancelled
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Step 2: Get compatible notebooks for selected service
    # For Kaggle, only show notebooks from kaggle/kernels/
    # For Colab, show notebooks from colab/ or jupyter_notebooks/
    all_notebooks = scanner.scan()

    if service == "kaggle":
        compatible_notebooks = [nb for nb in all_notebooks if nb.platform == "kaggle"]
        service_name = "Kaggle"
    elif service == "colab":
        compatible_notebooks = [nb for nb in all_notebooks if nb.platform in ("colab", "local")]
        service_name = "Colab"
    else:
        console.print(f"[red]Unknown service: {service}[/red]")
        return

    if not compatible_notebooks:
        console.print(f"[yellow]No compatible notebooks found for {service_name}[/yellow]")
        console.print(f"\nFor {service_name}, notebooks should be in:")
        if service == "kaggle":
            console.print("  • kaggle/kernels/*/notebook.ipynb")
        else:
            console.print("  • colab/*.ipynb")
            console.print("  • jupyter_notebooks/*.ipynb")
        return

    # Step 3: Select notebook (if not provided)
    nb = None
    if notebook is None:
        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            # Non-interactive: just use first compatible notebook
            if compatible_notebooks:
                nb = compatible_notebooks[0]
                console.print(f"[yellow]Non-interactive mode: selecting first notebook: {nb.name}[/yellow]")
            else:
                console.print("[red]No compatible notebooks found[/red]")
                return
        else:
            # Interactive mode: show selector
            # Create choices with rich information
            choices = []
            for idx, nb_info in enumerate(compatible_notebooks[:20], 1):  # Limit to 20 for UI
                # Calculate age
                from datetime import datetime
                age = datetime.now() - nb_info.modified
                if age.days > 0:
                    age_str = f"{age.days}d ago"
                elif age.seconds > 3600:
                    age_str = f"{age.seconds // 3600}h ago"
                elif age.seconds > 60:
                    age_str = f"{age.seconds // 60}m ago"
                else:
                    age_str = "just now"

                label = f"{nb_info.name[:40]:<40} [{nb_info.size_mb:>5.1f}MB] [{age_str:>10}]"
                choices.append(questionary.Choice(label, value=nb_info))

            selected_nb = questionary.select(
                f"Select notebook for {service_name}:",
                choices=choices,
                style=questionary.Style([
                    ('selected', 'bold green'),
                    ('pointer', 'bold green'),
                    ('highlighted', 'bold'),
                ])
            ).ask()

            if selected_nb is None:  # User cancelled
                console.print("[yellow]Cancelled[/yellow]")
                return

            nb = selected_nb
    else:
        # Find notebook by name
        nb = scanner.find_notebook(notebook)
        if not nb:
            console.print(f"[red]Notebook not found: {notebook}[/red]")
            return

        # Verify it's compatible
        if service == "kaggle" and nb.platform != "kaggle":
            console.print(f"[red]Notebook {nb.name} is not compatible with Kaggle[/red]")
            console.print(f"Platform: {nb.platform}, expected: kaggle")
            return
        elif service == "colab" and nb.platform not in ("colab", "local"):
            console.print(f"[red]Notebook {nb.name} is not compatible with Colab[/red]")
            console.print(f"Platform: {nb.platform}, expected: colab or local")
            return

    # Step 4: Extract config and confirm
    config = extract_notebook_config(nb.path)

    # Build config display
    config_lines = []
    if config:
        config_lines.append("\n[bold cyan]Config:[/bold cyan]")
        if 'use_sample' in config:
            sample_val = "[green]True[/green]" if config['use_sample'] else "[red]False[/red]"
            config_lines.append(f"  use_sample: {sample_val}")
        if 'sample_size' in config:
            config_lines.append(f"  sample_size: [yellow]{config['sample_size']}[/yellow]")
        if 'batch_size' in config:
            config_lines.append(f"  batch_size: [yellow]{config['batch_size']}[/yellow]")
        if 'epochs_stage1' in config and 'epochs_stage2' in config:
            config_lines.append(f"  epochs: [yellow]{config['epochs_stage1']} + {config['epochs_stage2']}[/yellow]")
        elif 'epochs_stage1' in config:
            config_lines.append(f"  epochs_stage1: [yellow]{config['epochs_stage1']}[/yellow]")
        elif 'epochs_stage2' in config:
            config_lines.append(f"  epochs_stage2: [yellow]{config['epochs_stage2']}[/yellow]")

    config_str = '\n'.join(config_lines) if config_lines else ""

    panel = Panel(
        f"[bold]{nb.relative_path}[/bold]\n"
        f"Platform: [magenta]{nb.platform}[/magenta]\n"
        f"Size: [green]{nb.size_mb:.1f}MB[/green]\n"
        f"Target: [cyan]{service_name}[/cyan]\n"
        f"Dry run: [yellow]{dry_run}[/yellow]"
        f"{config_str}",
        title="📤 Ready to Push",
        border_style="cyan"
    )
    console.print(panel)

    # Confirm (unless bypassed with CLI args or non-interactive)
    # Only confirm if: interactive terminal AND user didn't provide both parameters
    if sys.stdin.isatty() and not (user_provided_service and user_provided_notebook):
        confirm = questionary.confirm(
            "Push this notebook?",
            default=True
        ).ask()

        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            logger.log(
                action="select",
                status="cancelled",
                notebook=nb.relative_path,
                service=service
            )
            return

    # Push
    pusher = CloudPusher()
    result = pusher.push(nb, service, dry_run=dry_run)

    if result.get("success"):
        if dry_run:
            console.print("\n[yellow]🔍 Dry run - would execute:[/yellow]")
            console.print(f"[dim]{result.get('command')}[/dim]")
        else:
            console.print("\n[green]✅ Push successful![/green]")

            if result.get("stdout"):
                console.print("\n[dim]Output:[/dim]")
                console.print(result["stdout"])
    else:
        console.print(f"\n[red]❌ Push failed: {result.get('error')}[/red]")

        if result.get("stderr"):
            console.print("\n[dim]Error output:[/dim]")
            console.print(result["stderr"])

        raise typer.Exit(1)


@app.command()
def info():
    """Show CLI information and configuration."""
    scanner = NotebookScanner()
    logger = get_logger()

    console.print(Panel(
        f"[bold cyan]nbpush[/bold cyan] - Notebook Cloud Push Tool\n\n"
        f"[bold]Project Root:[/bold] {scanner.project_root}\n"
        f"[bold]Log File:[/bold] {logger.log_file}\n\n"
        f"[bold cyan]Scan Directories:[/bold cyan]\n" +
        "\n".join(f"  {dir} [{platform}]" for dir, platform in scanner.scan_dirs.items()),
        title="ℹ️  CLI Information",
        border_style="blue"
    ))

    logger.log(action="info", status="info")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
