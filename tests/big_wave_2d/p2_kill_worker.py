"""Helper process used by test_p2.py to simulate abrupt pass termination."""

import os
from pathlib import Path
import sys

import bfpy


def main() -> None:
    infile, outfile, scratch = map(Path, sys.argv[1:4])
    nx = int(sys.argv[4])
    ny = int(sys.argv[5])
    dtype = sys.argv[6]
    target_pass = sys.argv[7]
    memory_budget = int(sys.argv[8])

    def progress(pass_name: str, completed: int, total: int) -> None:
        if pass_name == target_pass and completed == total:
            os._exit(90)

    function = bfpy.fft2_c8 if dtype == "c8" else bfpy.fft2_c16
    function(
        infile,
        outfile,
        scratch,
        nx,
        ny,
        progress,
        None,
        memory_budget,
    )


if __name__ == "__main__":
    main()
