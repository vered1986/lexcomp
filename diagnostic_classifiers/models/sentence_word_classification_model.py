import torch
import numpy

import torch.nn.functional as F

from overrides import overrides
from typing import Dict, Optional

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.common.checks import check_dimensions_match
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder


@Model.register("sentence_word")
class SentenceWordClassificationModel(Model):
    """
    This ``Model`` performs classification given an in-context and out-of-context word representations.
    It can be used to answer questions like "is the given word in this sentence?", "it is used literally in the
    sentence?", etc.

    We embed the sentence, outputting a vector for each word in the sentence, and take the vector of the target word
    in the sentence, concatenate it with its out-of-context word embedding, and feed it to an MLP for classification.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    classifier_feedforward : ``FeedForward``
    sentence_encoder : ``Seq2VecEncoder``, optional (default=``None``)
        The encoder that we will use to convert the sentence to a vector.
    span_extractor: ``SpanExtractor``
        If provided, will attend the target vector to the rest of the vectors.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 span_extractor: Optional[SpanExtractor] = None,
                 sentence_encoder: Optional[Seq2SeqEncoder] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentenceWordClassificationModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_feedforward = classifier_feedforward
        self.sentence_encoder = sentence_encoder
        self.span_extractor = span_extractor

        # biLSTM
        if sentence_encoder:
            check_dimensions_match(text_field_embedder.get_output_dim(), sentence_encoder.get_input_dim(),
                                   "text embedding dim", "sentence_encoder input dim")

            check_dimensions_match(sentence_encoder.get_output_dim() + text_field_embedder.get_output_dim(),
                                   classifier_feedforward.get_input_dim(),
                                   "sentence_encoder + text_field_embedder output dim",
                                   "classifier_feedforward input dim")

        # Attention
        elif span_extractor:
            check_dimensions_match(text_field_embedder.get_output_dim() + span_extractor.get_output_dim(),
                                   classifier_feedforward.get_input_dim(),
                                   "text embedding dim + span_extractor", "classifier_feedforward input dim")

        # Only embedder
        else:
            check_dimensions_match(2 * text_field_embedder.get_output_dim(), classifier_feedforward.get_input_dim(),
                                   "2 * text embedding dim", "classifier_feedforward input dim")

        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()

        InitializerApplicator()(self)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                target_index: torch.LongTensor,
                target_word: Dict[str, torch.LongTensor],
                span_field: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentence : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        target_index : torch.LongTensor, required
            The index of the target word in the sentence.
        target_word : torch.LongTensor, required
            The target word in the sentence.
        span_field: The span field for the target word
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        ooc_word = self.text_field_embedder(target_word).squeeze(1)
        embedded_sentence = self.text_field_embedder(sentence)

        # Encode the tokens with a biLSTM
        if self.sentence_encoder:
            sentence_mask = util.get_text_field_mask(sentence)
            encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask).data
            batch_size, sequence_length, emb_dim = encoded_sentence.size()
            target_index_list = target_index.unsqueeze(2).repeat(1, 1, emb_dim)
            in_context_word = encoded_sentence.gather(1, target_index_list).squeeze(1)

        # Attend to the target vectors to the other tokens
        elif self.span_extractor:
            in_context_word = self.span_extractor(embedded_sentence, span_field).squeeze(0)

        # Just take the target vector
        else:
            batch_size, sequence_length, emb_dim = embedded_sentence.size()
            target_index_list = target_index.unsqueeze(2).repeat(1, 1, emb_dim)
            in_context_word = embedded_sentence.gather(1, target_index_list).squeeze(1)

        input = torch.cat([in_context_word, ooc_word], dim=-1)
        logits = self.classifier_feedforward(input)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}