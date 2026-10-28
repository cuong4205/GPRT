"""Compatibility entry point for the GP-only mode in the unified runner."""

import sys

from run_gprt_experiment import main


if __name__ == "__main__":
    if "--methods" not in sys.argv[1:]:
        sys.argv[1:1] = ["--methods", "gp"]
    main()
