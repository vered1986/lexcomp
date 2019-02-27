import json
import logging

from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.fields import LabelField, TextField, SpanField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("sentence_span_sentence_classification")
class SentenceSpanSentenceClassificationDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing a sentence with a target span, another sentence, and a label.
    Expected format for each input line: {"sentence": "text", "start": "int", "end": int, "label": "text"}
    and another field "paraphrase" or "relation" for the second sentence.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence1: ``TextField``
        sentence2: ``TextField``
        span: ``SpanField``
        span_text: ``TextField``
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words. Defaults to ``WordTokenizer()``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                curr_example_json = json.loads(line)
                sentence1 = curr_example_json['sentence']
                target_start = curr_example_json['start']
                target_end = curr_example_json['end']
                sentence2 = curr_example_json['paraphrase'] \
                    if 'paraphrase' in curr_example_json else curr_example_json['relation']
                label = curr_example_json['label']
                yield self.text_to_instance(sentence1, sentence2, target_start, target_end, label)

    @overrides
    def text_to_instance(self, sentence1: str, sentence2: str, start: int, end: int, label: str = None) -> Instance:
        tokenized_sentence1 = self._tokenizer.tokenize(sentence1)
        sentence_field1 = TextField(tokenized_sentence1, self._token_indexers)
        span_field1 = SpanField(start, end, sentence_field1)

        span_text = ' '.join(sentence1.split()[start:end+1])
        tokenized_span1 = self._tokenizer.tokenize(span_text)
        span_text_field1 = TextField(tokenized_span1, self._token_indexers)

        tokenized_sentence2 = self._tokenizer.tokenize(sentence2)
        sentence_field2 = TextField(tokenized_sentence2, self._token_indexers)
        span_field2 = SpanField(0, len(tokenized_sentence2) - 1, sentence_field2)

        fields = {'sentence1': sentence_field1, 'span_field1': span_field1, 'span1_text': span_text_field1,
                  'sentence2': sentence_field2, 'span_field2': span_field2}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)