import torch
import logging
import torch.nn.functional as F

from overrides import overrides
from typing import Dict, Optional
from torch.nn.modules import Linear

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.common.checks import check_dimensions_match
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

logger = logging.getLogger(__name__)



@Model.register("seq_label")
class SeqLabeler(Model):
    """
    This model performs sequence tagging to BIO tags.

    This implementation encodes the sentence and uses the vector of each word in the sentence
    to predict the BIO tag.

    Based on:
    https://github.com/allenai/allennlp/blob/335d8996a0ab6a2a3ea6ce323f57d2e76d9ddf1a/allennlp/models/semantic_role_labeler.py

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    sentence_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Optional[Seq2SeqEncoder] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SeqLabeler, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=['SENT_BOUND'])

        # Metrics just for specific classes
        all_labels = self.vocab.get_index_to_token_vocabulary("labels").values()
        logger.info(f'Labels: {all_labels}')
        mwe_labels = [label.replace('B-', '').replace('I-', '') for label in all_labels if 'MWE' in label]

        self.mwe_span_metric = None
        if len(mwe_labels) > 0:
            logger.info(f'Instantiating MWE only evaluation: {mwe_labels}')
            self.mwe_span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels",
                                                      ignore_classes=ne_labels + ['SENT_BOUND'])
        else:
            logger.info('No MWE only evaluation instantiated.')

        self.sentence_encoder = sentence_encoder

        # Encode the sentence with a biLSTM / self-attention
        if sentence_encoder:
            check_dimensions_match(text_field_embedder.get_output_dim(), sentence_encoder.get_input_dim(),
                                   "text embedding dim", "encoder input dim")
            self.tag_projection_layer = TimeDistributed(Linear(self.sentence_encoder.get_output_dim(), self.num_classes))

        # Just use embeddings
        else:
            self.tag_projection_layer = TimeDistributed(Linear(text_field_embedder.get_output_dim(), self.num_classes))

        InitializerApplicator()(self)

    def forward(self,  # type: ignore
                sentence_words: Dict[str, torch.LongTensor],
                sentence_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        sentence_words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        sentence_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_sentence = self.text_field_embedder(sentence_words)
        sentence_mask = get_text_field_mask(sentence_words)

        # Encode with biLSTM
        encoded_sentence = embedded_sentence
        if self.sentence_encoder:
            encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)

        logits = self.tag_projection_layer(encoded_sentence)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        batch_size, sequence_length, _ = embedded_sentence.size()
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if sentence_tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, sentence_tags, sentence_mask)

            for m in [m for m in [self.span_metric, self.mwe_span_metric] if m is not None]:
                m(class_probabilities, sentence_tags, sentence_mask)

            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = sentence_mask
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"sentence_tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['sentence_tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)

        # This can be a lot of metrics, as there are 3 per class.
        # we return the overall metrics with the 'O' label, and the labels only for named
        # entities and only for MWEs (if available).
        metric_dict_to_return = {x: y for x, y in metric_dict.items() if "overall" in x}

        if self.mwe_span_metric is not None:
            metric_dict_to_return['f1-MWE'] = self.mwe_span_metric.get_metric(reset=reset)['f1-measure-overall']

        return metric_dict_to_return

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

