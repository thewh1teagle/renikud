import torch
from torch import nn
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel
from typing import List, Optional, Union, Tuple


from .dicta_model import (
    is_hebrew_letter,
    remove_nikkud,
)


def remove_nikud(text: str):
    """Remove all nikud diacritics from Hebrew text"""
    return remove_nikkud(text)


@dataclass
class RenikudLogitsOutput(ModelOutput):
    """Output from RenikudModel with 3 separate prediction heads"""
    vowel_logits: torch.FloatTensor = None  # Shape: (batch, seq_len, 7)
    dagesh_logits: torch.FloatTensor = None  # Shape: (batch, seq_len, 2)
    sin_logits: torch.FloatTensor = None  # Shape: (batch, seq_len, 2)

    def detach(self):
        return RenikudLogitsOutput(
            self.vowel_logits.detach(),
            self.dagesh_logits.detach(),
            self.sin_logits.detach(),
        )


@dataclass
class ModelPredictions:
    """Container for all model predictions to avoid tuple unpacking."""
    vowel: List[List[int]]  # Class index 0-6
    dagesh: List[List[int]]  # Binary 0 or 1
    sin: List[List[int]]  # Binary 0 or 1 (0=shin, 1=sin)


class RenikudModel(BertPreTrainedModel):
    """Brand new diacritization model with 3 separate heads:
    - Vowel head: 7-class classifier (6 vowels + empty)
    - Dagesh head: Binary classifier (for בכךפףו only)
    - Sin head: Binary classifier (for ש only)
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # BERT backbone
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Dropout
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 3 separate prediction heads
        self.vowel_head = nn.Linear(config.hidden_size, 7)  # 6 vowels + empty
        self.dagesh_head = nn.Linear(config.hidden_size, 2)  # binary
        self.sin_head = nn.Linear(config.hidden_size, 2)  # binary
        
        # Initialize weights
        self.post_init()
        
        # Store vowel classes in config if not present
        if not hasattr(config, 'vowel_classes'):
            from constants import VOWEL_CLASSES
            config.vowel_classes = VOWEL_CLASSES

    def freeze_base_model(self):
        """Freeze BERT backbone, train only the 3 new heads"""
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], RenikudLogitsOutput]:
        """Forward pass through BERT and 3 prediction heads"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass through BERT
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = bert_outputs[0]  # last_hidden_state
        hidden_states = self.dropout(hidden_states)
        
        # Run through our 3 heads
        vowel_logits = self.vowel_head(hidden_states)
        dagesh_logits = self.dagesh_head(hidden_states)
        sin_logits = self.sin_head(hidden_states)
        
        if not return_dict:
            return (vowel_logits, dagesh_logits, sin_logits)
        
        return RenikudLogitsOutput(
            vowel_logits=vowel_logits,
            dagesh_logits=dagesh_logits,
            sin_logits=sin_logits,
        )

    def encode(
        self, sentences: list[str], tokenizer: BertTokenizerFast, padding="longest"
    ):
        """Tokenize sentences and return inputs with offset mapping"""
        sentences = [remove_nikud(sentence) for sentence in sentences]

        # Assert the lengths are within the tokenizer's max limit
        assert all(
            len(sentence) + 2 <= tokenizer.model_max_length for sentence in sentences
        ), (
            f"All sentences must be <= {tokenizer.model_max_length}, please segment and try again"
        )

        # Tokenize the inputs and return the tensor format
        inputs = tokenizer(
            sentences,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs, offset_mapping

    def decode(
        self,
        sentences,
        offset_mapping,
        vowel_predictions,
        dagesh_predictions,
        sin_predictions,
    ):
        """Reconstruct sentences with diacritics from predictions"""
        from constants import CAN_HAVE_DAGESH, CAN_HAVE_SIN_DOT
        
        ret = []
        for sent_idx, (sentence, sent_offsets) in enumerate(
            zip(sentences, offset_mapping)
        ):
            output = []
            prev_index = 0
            for idx, offsets in enumerate(sent_offsets):
                # Add anything missed
                if offsets[0] > prev_index:
                    output.append(sentence[prev_index : offsets[0]])
                if offsets[1] - offsets[0] != 1:
                    continue

                # Get the next character
                char = sentence[offsets[0] : offsets[1]]
                prev_index = offsets[1]

                if not is_hebrew_letter(char):
                    output.append(char)
                    continue

                # Apply the predictions to the character
                vowel = self.config.vowel_classes[vowel_predictions[sent_idx][idx]]
                
                # Apply dagesh only to letters that can have it
                dagesh = ""
                if char in CAN_HAVE_DAGESH and dagesh_predictions[sent_idx][idx] == 1:
                    from constants import DAGESH
                    dagesh = DAGESH
                
                # Apply sin/shin dot only to ש
                sin = ""
                if char in CAN_HAVE_SIN_DOT:
                    from constants import SIN_DOT
                    SHIN_DOT = '\u05c1'
                    sin = SIN_DOT if sin_predictions[sent_idx][idx] == 1 else SHIN_DOT

                output.append(char + sin + dagesh + vowel)

            output.append(sentence[prev_index:])
            ret.append("".join(output))

        return ret

    def get_predictions_from_output(self, output: RenikudLogitsOutput) -> ModelPredictions:
        """Extract predictions from an existing forward pass output"""
        # Get predictions from logits
        vowel_predictions = output.vowel_logits.argmax(dim=-1).tolist()
        dagesh_predictions = output.dagesh_logits.argmax(dim=-1).tolist()
        sin_predictions = output.sin_logits.argmax(dim=-1).tolist()

        return ModelPredictions(
            vowel=vowel_predictions,
            dagesh=dagesh_predictions,
            sin=sin_predictions,
        )

    def create_predictions(self, inputs) -> ModelPredictions:
        """Run forward pass and extract predictions"""
        output = self.forward(**inputs)
        return self.get_predictions_from_output(output)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        tokenizer: BertTokenizerFast,
        padding="longest",
    ):
        """Full prediction pipeline: encode -> forward -> decode"""
        # Step 1: Encoding (tokenizing sentences)
        inputs, offset_mapping = self.encode(sentences, tokenizer, padding)

        # Step 2: Making predictions
        predictions = self.create_predictions(inputs)

        # Step 3: Decoding (reconstructing the sentences with predictions)
        result = self.decode(
            sentences,
            offset_mapping,
            predictions.vowel,
            predictions.dagesh,
            predictions.sin,
        )

        return result
