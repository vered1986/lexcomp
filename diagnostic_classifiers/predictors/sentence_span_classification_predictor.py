from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentence_span_classification')
class SentenceSpanClassificationPredictor(Predictor):
    """"Predictor wrapper for the SentenceSpanClassificationModel"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        start = json_dict['start']
        end = json_dict['end']
        instance = self._dataset_reader.text_to_instance(sentence=sentence,
                                                         start=start,
                                                         end=end)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        _ = [label_dict[i] for i in range(len(label_dict))]
        return instance