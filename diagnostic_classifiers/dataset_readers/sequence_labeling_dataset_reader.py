import json
import logging

from typing import Dict, List
from overrides import overrides

from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


logger = logging.getLogger(__name__)


@DatasetReader.register("seq_label")
class SeqLabelReader(DatasetReader):
    """
    This DatasetReader is designed to read a sequence of words and tags.
    Expected format for each input line: {"sentence_words": "list", "sentence_tags": "list"}

    It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    tags : ``SequenceLabelField``
        A sequence of tags for the given word in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Default is ``{"tokens": SingleIdTokenIndexer()}``.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for MWE extraction and identification.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                curr_example_json = json.loads(line)
                sentence_words = curr_example_json['sentence_words']
                sentence_tags = curr_example_json['sentence_tags']
                yield self.text_to_instance(sentence_words, sentence_tags)


    @overrides
    def text_to_instance(self, sentence_words: List[Token], sentence_tags: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        sentence_field = TextField([Token(t) for t in sentence_words], self._token_indexers)
        fields['sentence_words'] = sentence_field

        if sentence_tags:
            fields['sentence_tags'] = SequenceLabelField(sentence_tags, sentence_field)
        return Instance(fields)