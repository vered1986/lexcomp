from overrides import overrides

from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize


@Predictor.register('seq_label')
class SeqLabelerPredictor(Predictor):
    """"Predictor wrapper for the SeqLabelerModel"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence_words = json_dict['sentence_words']
        instance = self._dataset_reader.text_to_instance(sentence_words=sentence_words)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        _ = [label_dict[i] for i in range(len(label_dict))]
        return instance

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs.pop("logits")
        outputs.pop("class_probabilities")
        return sanitize(outputs)