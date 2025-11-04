# nbpush CLI - Notebook Cloud Push Tool

**Version**: 1.0.0
**Purpose**: Push Jupyter notebooks to cloud GPU services with automatic activity logging

## Overview

`nbpush` is a command-line tool that streamlines pushing notebooks to cloud GPU services (Kaggle, Colab). It automatically scans your project for notebooks, provides an interactive interface, and logs all activity in an LLM-friendly format for your AI coding assistant.

## Features

- 📋 **Smart Notebook Discovery**: Automatically scans project directories
- 🕐 **Recency Sorting**: Shows most recently modified notebooks first
- 🚀 **Multi-Platform**: Push to Kaggle or Colab
- 📊 **Activity Logging**: All actions logged in JSON Lines format for LLM consumption
- 🎨 **Rich Interface**: Beautiful terminal UI with colors and tables
- 🔍 **Dry Run Mode**: Preview commands before execution

## Installation

```bash
./scripts/nbpush_install.sh
```

This will:
- Install dependencies (typer, rich)
- Create `./nbpush` executable in project root
- Set up activity logging

## Quick Start

```bash
# Interactive workflow (easiest)
./nbpush select

# List available notebooks
./nbpush list

# View activity
./nbpush activity
```

## Usage

### Interactive Selection (New!)

The `select` command provides a beautiful interactive UI:

```bash
# Full interactive workflow
./nbpush select

# Skip service selection
./nbpush select --service kaggle

# Skip all prompts (for scripts)
./nbpush select --service kaggle --notebook 07_transfer_learning --dry-run
```

**Features:**
- Beautiful service selection (Kaggle/Colab)
- Filtered notebook list (only compatible notebooks shown)
- Rich display with size, modification time, platform
- Auto-detects non-interactive mode (for scripts/automation)
- Optional confirmation before push
- CLI parameters bypass interactive prompts

### List Notebooks

```bash
# List all notebooks (sorted by most recent)
./nbpush list

# List only Kaggle notebooks
./nbpush list --platform kaggle

# List only local notebooks
./nbpush list --platform local

# Limit results
./nbpush list --limit 10
```

**Example Output:**
```
Notebooks (15 found)
┌─────┬─────────────────────────────────────────┬──────────┬────────┬──────────┐
│ #   │ Notebook                                │ Platform │   Size │ Modified │
├─────┼─────────────────────────────────────────┼──────────┼────────┼──────────┤
│ 1   │ colab/07_transfer_learning_colab.ipynb  │  colab   │  0.0MB │   1h ago │
│ 2   │ kaggle/kernels/07_transfer_learning/…   │  kaggle  │  0.0MB │   2h ago │
│ 3   │ jupyter_notebooks/07_transfer_learning… │  local   │  0.0MB │   2h ago │
└─────┴─────────────────────────────────────────┴──────────┴────────┴──────────┘
```

### Push to Cloud

#### Method 1: Interactive Selection (Recommended)

```bash
# Interactive workflow - select service then notebook
./nbpush select

# Or specify service, then select notebook interactively
./nbpush select --service kaggle

# Or specify both (no prompts, direct push)
./nbpush select --service kaggle --notebook 07_transfer_learning --dry-run
```

#### Method 2: Direct Push

```bash
# Push to Kaggle (dry run first)
./nbpush push <notebook> --service kaggle --dry-run

# Actually push to Kaggle
./nbpush push <notebook> --service kaggle

# Push to Colab
./nbpush push <notebook> --service colab
```

**Notebook Selection:**
- Use notebook name: `07_transfer_learning`
- Use relative path: `kaggle/kernels/07_transfer_learning/notebook.ipynb`
- Use partial match: `07_trans` (finds first match)

**Example:**
```bash
# Dry run
$ ./nbpush push 07_transfer --service kaggle --dry-run

╭────────────────────────────── 📤 Push Notebook ──────────────────────────────╮
│ kaggle/kernels/07_transfer_learning/notebook.ipynb                           │
│ Platform: kaggle                                                             │
│ Size: 0.0MB                                                                  │
│ Target: kaggle                                                               │
│ Dry run: True                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

🔍 Dry run - would execute:
cd /Volumes/SSD/Capstone/kaggle/kernels/07_transfer_learning &&
KAGGLE_CONFIG_DIR=../../../.kaggle kaggle kernels push
```

### View Activity

```bash
# Show recent activity
./nbpush activity

# Show summary statistics
./nbpush activity --summary

# Limit entries
./nbpush activity --limit 10
```

**Example Output:**
```
Recent Activity (10 entries)
┌─────────────────────┬────────┬─────────────────────────┬─────────┬───────────┐
│ Time                │ Action │ Notebook                │ Service │ Status    │
├─────────────────────┼────────┼─────────────────────────┼─────────┼───────────┤
│ 2025-11-04T19:54:23 │ push   │ ...s/07_transfer_learn… │ kaggle  │ completed │
│ 2025-11-04T19:54:18 │ list   │ -                       │ -       │ completed │
│ 2025-11-04T19:54:13 │ push   │ ..._notebooks/07_trans… │ kaggle  │ failed    │
└─────────────────────┴────────┴─────────────────────────┴─────────┴───────────┘
```

### Show Configuration

```bash
./nbpush info
```

Shows:
- Project root
- Log file location
- Scanned directories

## Activity Logging

All CLI activity is logged to `.logs/nbpush_activity.jsonl` in JSON Lines format.

### Log Format

Each action is a single-line JSON entry:

```json
{"timestamp":"2025-11-04T19:54:23.194520Z","timestamp_local":"2025-11-04T19:54:23.194666","action":"push","status":"started","notebook":"kaggle/kernels/07_transfer_learning/notebook.ipynb","service":"kaggle","metadata":{"dry_run":true}}
```

### Log Fields

- `timestamp`: UTC timestamp (ISO 8601)
- `timestamp_local`: Local timestamp (ISO 8601)
- `action`: Action type (`list`, `push`, `activity`, `info`)
- `status`: Status (`started`, `completed`, `failed`, `info`)
- `notebook`: Notebook path (if applicable)
- `service`: Cloud service (`kaggle`, `colab`, or null)
- `metadata`: Additional data (command, counts, errors)
- `error`: Error message (if failed)

### Why JSON Lines?

JSON Lines (`.jsonl`) format is optimized for:
- **LLM Consumption**: Easy to parse and analyze
- **Streaming**: Can process line-by-line
- **Append-Only**: No need to parse entire file
- **Structured**: Full context in each line

Your AI coding assistant (Claude) can easily read this log to understand:
- What notebooks you're working on
- Which platforms you're using
- Success/failure patterns
- Recent activity trends

## Platform Configuration

The tool scans these directories:

| Directory              | Platform | Purpose                          |
|------------------------|----------|----------------------------------|
| `jupyter_notebooks/`   | local    | Local development notebooks      |
| `kaggle/kernels/`      | kaggle   | Kaggle kernel configurations     |
| `colab/`               | colab    | Google Colab notebooks           |

## Requirements

### For Kaggle Push

- Notebook must be in `kaggle/kernels/<name>/` directory
- Must have `kernel-metadata.json` in same directory
- Kaggle API configured in `.kaggle/kaggle.json`

### For Colab Push

- `colab-cli` must be installed
- OAuth credentials configured (`.colab/client_secrets.json`)

## Examples

### Workflow Example 1: Push to Kaggle

```bash
# 1. List notebooks to see what's available
./nbpush list --platform kaggle

# 2. Dry run to verify command
./nbpush push 07_transfer --service kaggle --dry-run

# 3. Actually push
./nbpush push 07_transfer --service kaggle

# 4. Check activity log
./nbpush activity
```

### Workflow Example 2: Quick Push

```bash
# Find and push in one go (uses fuzzy matching)
./nbpush push 07_trans --service kaggle
```

### Workflow Example 3: Check Recent Activity

```bash
# See what you've been working on
./nbpush activity --summary

# Output:
# Total Actions: 15
# Last Activity: 2025-11-04T19:54:30
#
# Actions:
#   list: 6
#   push: 8
#   activity: 1
#
# Status:
#   completed: 10
#   failed: 2
#   started: 3
```

## Troubleshooting

### Error: "Notebook not found"

The tool uses fuzzy matching, but you can be more specific:
- Use full relative path: `kaggle/kernels/07_transfer_learning/notebook.ipynb`
- Check available notebooks: `./nbpush list`

### Error: "No kernel-metadata.json found"

Kaggle notebooks need `kernel-metadata.json` in the same directory:
```bash
cd kaggle/kernels/07_transfer_learning
ls kernel-metadata.json  # Should exist
```

### Error: "colab-cli not found"

Install colab-cli:
```bash
pip install colab-cli
```

## Integration with AI Assistant

The activity log is designed for your AI coding assistant (Claude). When you run:

```bash
./nbpush push 07_transfer --service kaggle
```

Claude can later read `.logs/nbpush_activity.jsonl` to see:
- You pushed `07_transfer_learning` to Kaggle
- When you pushed it
- Whether it succeeded
- What command was executed

This gives Claude full context of your cloud GPU workflow without you having to explain it.

## Tips

1. **Use dry-run first**: Always test with `--dry-run` to verify the command
2. **Check logs regularly**: `./nbpush activity` shows what you've been doing
3. **Use fuzzy matching**: Type partial names like `07_trans` instead of full paths
4. **Filter by platform**: `--platform kaggle` to see only relevant notebooks
5. **Add to PATH**: For convenience, add project root to PATH

## Future Enhancements

Potential additions:
- `nbpush status` - Check cloud job status
- `nbpush download` - Download results
- `nbpush cancel` - Cancel running jobs
- Interactive selection (arrow keys)
- Config file for custom directories

## Technical Details

**Architecture:**
```
nbpush/
├── cli.py       # Typer CLI interface
├── scanner.py   # Notebook discovery and sorting
├── pusher.py    # Cloud push handlers (Kaggle/Colab)
└── logger.py    # JSON Lines activity logging
```

**Dependencies:**
- `typer` - CLI framework
- `rich` - Terminal UI
- Python 3.11+

**Log Location:** `.logs/nbpush_activity.jsonl`

## License

Part of NIH Chest X-Ray project. See project LICENSE.
