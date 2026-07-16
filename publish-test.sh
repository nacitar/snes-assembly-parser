#!/bin/bash
set -eu
rm -rf dist
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev$(date +%s)" uv build
uv run twine check dist/*
uv run twine upload --skip-existing --repository testpypi dist/*
