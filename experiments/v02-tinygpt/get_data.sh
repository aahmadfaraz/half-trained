#!/bin/bash
# Tiny Shakespeare (public domain text), as packaged in karpathy/char-rnn
cd "$(dirname "$0")" && mkdir -p data
curl -sSL -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  data/tinyshakespeare.txt" | shasum -a 256 -c
