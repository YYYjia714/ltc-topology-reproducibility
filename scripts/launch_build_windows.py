from __future__ import annotations

import subprocess
import runpy
import sys
from pathlib import Path


def _resolve_progress_file(argv: list[str], project_root: Path) -> Path:
    if "--progress-file" in argv:
        index = argv.index("--progress-file")
        if index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser()
    return project_root / "runs" / "progress" / "window_progress.json"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    argv = sys.argv[1:]
    open_monitor = True
    if "--no-monitor" in argv:
        argv = [item for item in argv if item != "--no-monitor"]
        open_monitor = False
    progress_file = _resolve_progress_file(argv, project_root)
    if open_monitor:
        subprocess.Popen(
            [
                sys.executable,
                str(project_root / "monitor_progress.py"),
                "--progress-file",
                str(progress_file),
            ],
            cwd=project_root,
        )
    sys.path.insert(0, str(project_root))
    sys.argv = [str(project_root / "scripts" / "build_forecasting_windows.py"), *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
