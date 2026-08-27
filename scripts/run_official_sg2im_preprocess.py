#!/usr/bin/env python3
"""Run the pinned SG2IM preprocessor with a modern SciPy compatibility shim."""

import argparse
import runpy
import sys

import imageio
import scipy.misc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocess_script")
    args, forwarded = parser.parse_known_args()

    # SG2IM imports these deprecated symbols although its graph encoding path
    # does not use them. Supplying the aliases preserves the official code.
    if not hasattr(scipy.misc, "imread"):
        scipy.misc.imread = imageio.imread
    if not hasattr(scipy.misc, "imresize"):
        scipy.misc.imresize = lambda image, size: imageio.imresize(image, size)

    sys.argv = [args.preprocess_script, *forwarded]
    runpy.run_path(args.preprocess_script, run_name="__main__")


if __name__ == "__main__":
    main()
