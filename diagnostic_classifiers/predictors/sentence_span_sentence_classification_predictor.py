from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentence_span_sentence_classification')
class SentenceSpanSentenceClassificationPredictor(Predictor):
    """"Predictor wrapper for the SentenceSpanSentenceClassificationModel"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence1 = json_dict['sentence']
        start = json_dict['start']
        end = json_dict['end']
        sentence2 = json_dict['paraphrase']
        instance = self._dataset_reader.text_to_instance(sentence1=sentence1,
                                                         sentence2=sentence2,
                                                         start=start,
                                                         end=end)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        _ = [label_dict[i] for i in range(len(label_dict))]
        return instance

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs['sentence1'] = ' '.join([t.text for t in instance['sentence1'].tokens])
        outputs['sentence2'] = ' '.join([t.text for t in instance['sentence2'].tokens])
        outputs['span_text'] = ' '.join([t.text for t in instance['span1_text'].tokens])
        return sanitize(outputs)
