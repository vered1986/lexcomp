import json
import logging

from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, IndexField, SpanField

logger = logging.getLogger(__name__)


@DatasetReader.register("sentence_word")
class SentenceWordDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sentences with a target word, and creates a dataset for their classification.

    Expected format for each input line: {"sentence": "text", "target_index": "int", "target_word": "text",
    "label": "text"}

    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``
        target_index: ``int``
        target_word: ``TextField``
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
                sentence = curr_example_json['sentence']
                target_index = curr_example_json['target_index']
                target_word = curr_example_json['target_word']
                label = curr_example_json['label']
                yield self.text_to_instance(sentence, target_index, target_word, label)

    @overrides
    def text_to_instance(self, sentence: str, target_index: int, target_word: str, label: str = None) -> Instance:
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        tokenized_target_word = self._tokenizer.tokenize(target_word)
        target_word_field = TextField(tokenized_target_word, self._token_indexers)
        target_index_field = IndexField(target_index, sentence_field)
        span_field = SpanField(target_index, target_index, sentence_field)

        fields = {'sentence': sentence_field, 'target_index': target_index_field,
                  'target_word': target_word_field, 'span_field': span_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)