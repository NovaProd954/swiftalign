"""
swiftalign.dashboard
~~~~~~~~~~~~~~~~~~~~
Rich-text console dashboard for SwiftAlign training runs.
Falls back to plain logging output when ``rich`` is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    pass


# ── Colour / style constants ──────────────────────────────────────────────────
_ACCENT   = "bold cyan"
_GOOD     = "bold green"
_WARN     = "bold yellow"
_BAD      = "bold red"
_DIM      = "dim white"
_HEADING  = "bold white"


class Dashboard:
    """
    Wraps the Rich Console and exposes high-level methods used by the runner.
    Gracefully degrades to stdlib logging when Rich is unavailable.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and _RICH_AVAILABLE
        self._console: Optional[object] = Console() if self.enabled else None
        self._start_time: float = time.time()
        self._progress = None
        self._task_id = None

    # ── Banner ──────────────────────────────────────────────────────────────

    def banner(self):
        if not self.enabled:
            logger.info("=" * 60)
            logger.info("  SwiftAlign — alignment training scaffold  v0.1.0")
            logger.info("=" * 60)
            return

        art = Text()
        art.append("  ███████╗██╗    ██╗██╗███████╗████████╗\n", style="bold cyan")
        art.append("  ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝\n", style="bold cyan")
        art.append("  ███████╗██║ █╗ ██║██║█████╗     ██║   \n", style="cyan")
        art.append("  ╚════██║██║███╗██║██║██╔══╝     ██║   \n", style="cyan")
        art.append("  ███████║╚███╔███╔╝██║██║        ██║   \n", style="dim cyan")
        art.append("  ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝   \n", style="dim cyan")
        art.append("  ALIGN\n", style="bold white")

        self._console.print(
            Panel(art, title="[bold cyan]SwiftAlign v0.1.0[/]",
                  subtitle="[dim]compact alignment training scaffold[/]",
                  border_style="cyan", padding=(0, 2))
        )

    # ── Hardware summary ─────────────────────────────────────────────────────

    def hardware_table(self, hw_summary: dict):
        if not self.enabled:
            for k, v in hw_summary.items():
                logger.info("  %-22s %s", k, v)
            return

        t = Table(title="[bold]Hardware Profile[/]", box=box.SIMPLE_HEAD,
                  show_header=True, header_style="bold cyan",
                  border_style="dim cyan")
        t.add_column("Property", style="cyan", no_wrap=True)
        t.add_column("Value", style="white")

        _icons = {
            "device": "🖥 ",
            "gpu": "⚡",
            "vram_gb": "💾",
            "dtype": "🔢",
            "attention": "👁 ",
            "qlora_rec": "🗜 ",
            "grad_ckpt": "♻ ",
        }
        for k, v in hw_summary.items():
            icon = _icons.get(k, "  ")
            style = _GOOD if str(v) in ("True", "cuda") else _DIM
            t.add_row(f"{icon} {k}", f"[{style}]{v}[/]")

        self._console.print(t)

    # ── Config summary ────────────────────────────────────────────────────────

    def config_panel(self, cfg: dict):
        if not self.enabled:
            logger.info("Run config: %s", cfg)
            return

        lines = "\n".join(
            f"  [cyan]{k:<22}[/] [white]{v}[/]"
            for k, v in cfg.items()
        )
        self._console.print(
            Panel(lines, title="[bold]Run Configuration[/]",
                  border_style="cyan", padding=(0, 1))
        )

    # ── Section headers ───────────────────────────────────────────────────────

    def section(self, title: str):
        if not self.enabled:
            logger.info("\n── %s ──", title)
            return
        self._console.print(Rule(f"[bold cyan]{title}[/]", style="cyan"))

    # ── Training progress ─────────────────────────────────────────────────────

    def start_progress(self, total_steps: int, description: str = "Training"):
        if not self.enabled:
            return

        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold cyan]{task.description}[/]"),
            BarColumn(bar_width=None, style="cyan", complete_style="bold cyan"),
            TextColumn("[cyan]{task.completed}/{task.total}[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            expand=True,
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task(description, total=total_steps)

    def update_progress(self, advance: int = 1, metrics: Optional[dict] = None):
        if not self.enabled or self._progress is None:
            if metrics:
                logger.info("  step metrics: %s", metrics)
            return

        desc_parts = ["Training"]
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    desc_parts.append(f"{k}=[yellow]{v:.4f}[/]")
                else:
                    desc_parts.append(f"{k}=[yellow]{v}[/]")
        self._progress.update(
            self._task_id,
            advance=advance,
            description=" | ".join(desc_parts),
        )

    def stop_progress(self):
        if not self.enabled or self._progress is None:
            return
        self._progress.__exit__(None, None, None)
        self._progress = None

    # ── Results table ─────────────────────────────────────────────────────────

    def results_table(self, metrics: dict):
        if not self.enabled:
            logger.info("Results: %s", metrics)
            return

        t = Table(title="[bold]Training Results[/]", box=box.ROUNDED,
                  show_header=True, header_style="bold green",
                  border_style="green")
        t.add_column("Metric", style="green", no_wrap=True)
        t.add_column("Value", style="white", justify="right")

        for k, v in metrics.items():
            if isinstance(v, float):
                t.add_row(k, f"{v:.6f}")
            else:
                t.add_row(k, str(v))

        self._console.print(t)

    # ── Done ──────────────────────────────────────────────────────────────────

    def done(self, output_dir: str):
        elapsed = time.time() - self._start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        if not self.enabled:
            logger.info("Training complete in %02d:%02d:%02d → %s", h, m, s, output_dir)
            return

        self._console.print(
            Panel(
                f"[bold green]✓ Training complete[/]  [dim]elapsed: {h:02d}:{m:02d}:{s:02d}[/]\n"
                f"[dim]Output:[/] [cyan]{output_dir}[/]",
                border_style="green", padding=(0, 2)
            )
        )

    # ── Generic log ──────────────────────────────────────────────────────────

    def log(self, message: str, style: str = "white"):
        if not self.enabled:
            logger.info(message)
            return
        self._console.print(f"[{style}]{message}[/]")

    def warn(self, message: str):
        if not self.enabled:
            logger.warning(message)
            return
        self._console.print(f"[{_WARN}]⚠  {message}[/]")

    def error(self, message: str):
        if not self.enabled:
            logger.error(message)
            return
        self._console.print(f"[{_BAD}]✗  {message}[/]")
