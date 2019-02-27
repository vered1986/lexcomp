import torch

import numpy as np
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


@Model.register("sentence_span_classification")
class SentenceSpanClassificationModel(Model):
    """
    This ``Model`` performs classification of sentence (with a given span of interest) to a label.

    We embed the sentence with the text_field_embedder, and possibly encode it.
    Then we apply an extractor to get the vectors associated with the span.

    We feed this vector into a FF network for classification.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    classifier_feedforward : ``FeedForward``
    span_extractor: ``SpanExtractor``
        If provided, will combine the span into one vector
    seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        The encoder that we will use to convert the sentence to a sequence of vectors.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 span_extractor: Optional[SpanExtractor] = None,
                 seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentenceSpanClassificationModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_feedforward = classifier_feedforward
        self.seq2seq_encoder = seq2seq_encoder
        self.span_extractor = span_extractor

        # Using an encoder that returns a vector for each token
        if seq2seq_encoder:
            check_dimensions_match(text_field_embedder.get_output_dim(), seq2seq_encoder.get_input_dim(),
                                   "text embedding dim", "seq2seq_encoder input dim")

            check_dimensions_match(seq2seq_encoder.get_output_dim(), span_extractor.get_input_dim(),
                                   "seq2seq_encoder output dim", "span_extractor input dim")

            check_dimensions_match(span_extractor.get_output_dim(), classifier_feedforward.get_input_dim(),
                                   "span_extractor", "classifier_feedforward input dim")

        # Using only an embedder, it will return a vector for each token, and it needs to go through a
        # span extractor afterwards.
        else:
            check_dimensions_match(text_field_embedder.get_output_dim(), span_extractor.get_input_dim(),
                                   "text embedding dim", "span_extractor input dim")

            check_dimensions_match(span_extractor.get_output_dim(), classifier_feedforward.get_input_dim(),
                                   "span_extractor", "classifier_feedforward input dim")

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()

        InitializerApplicator()(self)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                span: torch.LongTensor,
                span_text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentence : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        span : torch.LongTensor, required
            The span field
        span_text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
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
        embedded_sentence = self.text_field_embedder(sentence)

        # Encode the sequence
        if self.seq2seq_encoder:
            sentence_mask = util.get_text_field_mask(sentence)
            encoded_sentence = self.seq2seq_encoder(embedded_sentence, sentence_mask).data

        # Using an embedder that returns a vector for each token.
        # We take the span from it.
        else:
            encoded_sentence = embedded_sentence

        # Extract the span: shape = (batch_size, num_spans, feed_forward.input_dim())
        input = self.span_extractor(encoded_sentence, span).squeeze(0)

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
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        output_dict = super(SentenceSpanClassificationModel, self).decode(output_dict)
        label_probs = torch.nn.functional.softmax(output_dict['logits'], dim=-1)
        output_dict['label_probs'] = label_probs
        predictions = label_probs.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)

        # Single instance
        if np.isscalar(argmax_indices):
            argmax_indices = [argmax_indices]

        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}