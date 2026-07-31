#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import tempfile
from itertools import combinations_with_replacement
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "calibrate_dspark_sps.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        log = base / "run.log"
        output = base / "profile.conf"
        lines = [
            "ds4: DSpark offline SPS fingerprint=123456789abcdef0 "
            "profile=/tmp/profile.conf\n"
        ]
        for rows in range(2, 13):
            shapes = [
                shape
                for shape in combinations_with_replacement(range(1, 7), 2)
                if sum(shape) == rows
            ]
            for shape_index, shape in enumerate(shapes):
                for sample in range(10):
                    verify = (
                        10.0 + rows + shape_index * 0.5 + sample * 0.01
                    )
                    shape_text = ",".join(str(value) for value in shape)
                    lines.append(
                        "ds4-bench: SPS sample executor=physical "
                        "path=neural bucket=31 R=2 "
                        f"batch={rows} rows={rows} "
                        f"verify={verify:.3f} ms "
                        f"shape={shape_text} run={sample}\n"
                    )
        log.write_text("".join(lines), encoding="utf-8")
        completed = subprocess.run(
            [
                "python3",
                str(TOOL),
                "--log",
                str(log),
                "--output",
                str(output),
                "--min-samples",
                "8",
                "--skip-first",
                "0",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        text = output.read_text(encoding="ascii")
        assert "complete_groups=1" in completed.stdout
        assert text.startswith("DS4_DSPARK_SPS_V1 123456789abcdef0\n")
        assert sum(line.startswith("C ") for line in text.splitlines()) == 11
        row_b7 = next(
            line for line in text.splitlines()
            if line.startswith("C 1 0 31 2 7 ")
        )
        verify_seconds = float(row_b7.split()[6])
        # B=7 has [1,6], [2,5] and [3,4]. The last partition is deliberately
        # 1 ms slower, and the authoritative row must preserve that cost.
        assert abs(verify_seconds - 0.018045) < 1.0e-9

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        log = base / "anchors.log"
        output = base / "interpolated.conf"
        lines = [
            "ds4: DSpark offline SPS fingerprint=123456789abcdef0 "
            "profile=/tmp/profile.conf\n"
        ]
        for bucket, offset in ((0, 10.0), (2, 30.0)):
            for rows in range(1, 7):
                for sample in range(10):
                    verify = offset + rows + sample * 0.01
                    lines.append(
                        "ds4-bench: SPS sample executor=physical "
                        f"path=neural bucket={bucket} R=1 "
                        f"batch={rows} rows={rows} "
                        f"verify={verify:.3f} ms "
                        f"shape={rows} run={sample}\n"
                    )
        log.write_text("".join(lines), encoding="utf-8")
        completed = subprocess.run(
            [
                "python3",
                str(TOOL),
                "--log",
                str(log),
                "--output",
                str(output),
                "--min-samples",
                "8",
                "--skip-first",
                "0",
                "--interpolate-context",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        text = output.read_text(encoding="ascii")
        assert "measured=12 interpolated=6 complete_groups=3" in completed.stdout
        row_b1 = next(
            line for line in text.splitlines()
            if line.startswith("C 1 0 1 1 4 ")
        )
        assert abs(float(row_b1.split()[6]) - 0.024045) < 1.0e-9
        assert not any(
            line.startswith("C 1 0 3 1 ") for line in text.splitlines()
        )

    print("dspark SPS calibration tests: OK")


if __name__ == "__main__":
    main()
