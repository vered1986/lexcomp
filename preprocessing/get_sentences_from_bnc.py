# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('bnc_path', help='The path to the BNC corpus files')
ap.add_argument('dataset_path', help='The path to the dataset directory with the BNC IDs')
ap.add_argument('out_path', help='Where to save the dataset with the sentences from BNC')
args = ap.parse_args()

import logging

logger = logging.getLogger(__name__)

import os
import re
import json
import tqdm
import spacy
import codecs
import logging

logger = logging.getLogger(__name__)

import xml.etree.ElementTree as ET

from unidecode import unidecode

from allennlp.data.tokenizers import WordTokenizer


def main():
    bnc_reader = BNCDatasetReader(args.bnc_path)
    dataset = {}

    for s in ['train', 'test', 'val']:
        in_file = os.path.join(args.dataset_path, f'ids_{s}.jsonl')
        logger.info(f'Reading from {in_file}')
        dataset[s] = []

        with codecs.open(in_file, 'r', 'utf-8') as f_in:
            for line in tqdm.tqdm(f_in):
                try:
                    curr_example_json = json.loads(line.strip())
                    instance = bnc_reader.get_single_instance_from_json(curr_example_json)
                    if instance is not None:
                        dataset[s].append(instance)
                except:
                    logger.warning(f'Error in line: {line}')
                    pass

    for s in ['train', 'test', 'val']:
        out_file = os.path.join(args.out_path, f'{s}.jsonl')
        logger.info(f'Writing the to {out_file}')

        with codecs.open(out_file, 'w', 'utf-8') as f_out:
            for instance in dataset[s]:
                f_out.write(json.dumps(instance) + '\n')



class BNCDatasetReader:
    """
    Reads sentences from the BNC corpus given a sentence ID.
    If you didn't download the BNC corpus yet, please do so.
    """
    def __init__(self, bnc_corpus_path):
        self._tokenizer = WordTokenizer()
        self.bnc_corpus_path = bnc_corpus_path
        self.nlp = spacy.load('en')

    def get_single_instance_from_json(self, curr_example_json):
        """
        Reads a sentence from the BNC corpus from its ID and returns an item or None if invalid.
        """
        instance = None
        sentence_id = curr_example_json['bnc_id']

        # Get the sentence. The format of the BNCID is: BNC_file_dir/BNC_xml_fileName/sentence_number
        items = sentence_id.split('/')
        curr_sentence_id = items[-1]
        curr_sentence_file_path = '/'.join(items[:-1])
        tree = ET.parse(os.path.join(self.bnc_corpus_path, curr_sentence_file_path))
        root = tree.getroot()
        sentence = ''
        for element in root.findall(".//s[@n='{}']".format(curr_sentence_id)):
            sentence = ''.join(element.itertext())

        if sentence != '':
            try:
                # Remove unicode punctuation
                sentence = re.sub('\s+', ' ', unidecode(sentence)).lower()
                span_text = curr_example_json['span_text']
                span_text = re.sub('\s+', ' ', unidecode(span_text)).lower()

                # Lemmatize the sentence
                tokens = [t for t in self.nlp(sentence)]
                lemmas = [t.lemma_ if t.lemma_ != '-PRON-' else t.lower_ for t in tokens]
                lemmas = [lemma if lemma != "n't" else "not" for lemma in lemmas]

                # Lemmatize the span.
                span_tokens = [t for t in self.nlp(span_text)]
                span_lemmas = [t.lemma_ if t.lemma_ != '-PRON-' else t.lower_ for t in span_tokens]
                span_lemmas = [lemma if lemma != "n't" else "not" for lemma in span_lemmas]

                # Find the span within the sentence
                index_within_sentence = [i for i in range(len(lemmas) - len(span_lemmas) + 1) if
                                         lemmas[i:i + len(span_lemmas)] == span_lemmas]

                if len(index_within_sentence) > 0:
                    start_token_index = index_within_sentence[0]
                    end_token_index = start_token_index + len(span_lemmas) - 1
                    label = curr_example_json['label']
                    tokenized_sentence = ' '.join([t.text for t in tokens])
                    assert(' '.join(lemmas[start_token_index:end_token_index+1]) == span_text)

                    instance = { 'bnc_id': sentence_id,
                                 'sentence': tokenized_sentence,
                                 'start': start_token_index,
                                 'end': end_token_index,
                                 'span_text': span_text,
                                 'label': label }
                else:
                    lemmatized_sent = ' '.join(lemmas)
                    lemmatized_span = '  '.join(span_lemmas)
                    logger.warning(f'Failed to find span. Sentence: "{sentence}", ' +
                                   f'sentence lemmatized: "{lemmatized_sent}", ' +
                                   f'span: "{span_text}", span lemmatized: "{lemmatized_span}"')
            except Exception as e:
                logger.warning(e)

        return instance


if __name__ == '__main__':
    main()
