from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def test_src_tree_has_no_stale_polyform_or_commercial_headers() -> None:
    bad_markers = (
        "PolyForm Noncommercial",
        "For commercial licensing inquiries:",
    )
    offenders: list[str] = []

    for path in sorted(SRC_DIR.rglob("*.py")):
        head = path.read_text(errors="ignore").splitlines()[:8]
        joined = "\n".join(head)
        if any(marker in joined for marker in bad_markers):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
