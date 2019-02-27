from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentence_word')
class SentenceWordPredictor(Predictor):
    """"Predictor wrapper for the SentenceWordClassificationModel"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        target_index = json_dict['target_index']
        target_word = json_dict['target_word']
        instance = self._dataset_reader.text_to_instance(sentence=sentence,
                                                         target_word=target_word,
                                                         target_index=target_index)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        _ = [label_dict[i] for i in range(len(label_dict))]
        return instance