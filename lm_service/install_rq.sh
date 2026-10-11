#!/bin/bash
set -e
pip install -r requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
python -m spacy download en_core_web_trf pl_core_news_lg
