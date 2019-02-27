#!/usr/bin/env bash

mkdir -p pretrained_models;

echo "Downloading fastText...";
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip;
unzip wiki-news-300d-1M-subword.vec.zip;

# Remove the first line indicating the size
sed '1d' wiki-news-300d-1M-subword.vec > pretrained_models/fasttext.txt;
rm wiki-news-300d-1M-subword.vec;
gzip pretrained_models/fasttext.txt;
