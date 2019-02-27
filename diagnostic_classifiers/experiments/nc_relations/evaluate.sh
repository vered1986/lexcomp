#!/bin/bash

declare -a word_embeddings=(word2vec glove fasttext)
declare -a contextual_embeddings=(elmo openai bert)
declare -a layers=(all top)
declare -a all_encodings=(noenc bilm)

for encoding in "${all_encodings[@]}"
do
    for embeddings in "${word_embeddings[@]}"
    do
        allennlp evaluate output/diagnostic_classifiers/nc_relations/${embeddings}/${encoding}/model.tar.gz \
        diagnostic_classifiers/data/nc_relations/test.jsonl \
        --output-file output/diagnostic_classifiers/nc_relations/${embeddings}/${encoding}/test_results.json \
        --include-package diagnostic_classifiers &
    done
    wait

    for embeddings in "${contextual_embeddings[@]}"
    do
        for layer in "${layers[@]}"
        do
            allennlp evaluate output/diagnostic_classifiers/nc_relations/${embeddings}_${layer}/${encoding}/model.tar.gz \
            diagnostic_classifiers/data/nc_relations/test.jsonl \
            --output-file output/diagnostic_classifiers/nc_relations/${embeddings}_${layer}/${encoding}/test_results.json \
            --include-package diagnostic_classifiers &
        done
        wait
    done
    wait
done
