from typing import List, Tuple
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
import re


@dataclass
class Batch:
    text: List[str]  # Unvocalized text (for tokenization)
    vocalized: List[str]  # Vocalized text with diacritics (for evaluation)
    input: BatchEncoding
    vowel_targets: torch.Tensor  # Shape: (batch, seq_len) - class indices
    dagesh_targets: torch.Tensor  # Shape: (batch, seq_len) - binary
    sin_targets: torch.Tensor  # Shape: (batch, seq_len) - binary


class TrainData(Dataset):
    """Dataset that extracts 3 labels per character: vowel, dagesh, sin"""
    
    def __init__(self, unvocalized_lines: List[str], vocalized_lines: List[str]):
        from constants import VOWEL_CLASSES, CAN_HAVE_DAGESH, CAN_HAVE_SIN_DOT, DAGESH, SIN_DOT
        
        self.unvocalized_lines = unvocalized_lines
        self.vocalized_lines = vocalized_lines
        self.vowel_classes = VOWEL_CLASSES
        self.can_have_dagesh = CAN_HAVE_DAGESH
        self.can_have_sin_dot = CAN_HAVE_SIN_DOT
        self.dagesh = DAGESH
        self.sin_dot = SIN_DOT
        self.shin_dot = '\u05c1'
        
        # Create mappings
        self.vowel_to_idx = {v: i for i, v in enumerate(VOWEL_CLASSES)}
        
        # Pattern to identify diacritics
        self.nikud_pattern = re.compile(r'[\u05b0-\u05bc\u05c1\u05c2\u05c7]')

    def __len__(self):
        return len(self.unvocalized_lines)

    def __getitem__(self, idx):
        unvocalized_line = self.unvocalized_lines[idx]
        vocalized_line = self.vocalized_lines[idx]
        
        # Extract labels for each character in the unvocalized text
        vowel_labels = []
        dagesh_labels = []
        sin_labels = []
        
        # Build a mapping from unvocalized position to diacritics
        char_to_diacritics = self._extract_diacritics(unvocalized_line, vocalized_line)
        
        for i, char in enumerate(unvocalized_line):
            diacritics = char_to_diacritics.get(i, [])
            
            # Extract vowel (first non-dagesh, non-sin diacritic)
            vowel = ''
            for d in diacritics:
                if d in self.vowel_classes and d != '':
                    vowel = d
                    break
            vowel_labels.append(self.vowel_to_idx.get(vowel, 0))
            
            # Extract dagesh
            has_dagesh = self.dagesh in diacritics and char in self.can_have_dagesh
            dagesh_labels.append(1 if has_dagesh else 0)
            
            # Extract sin/shin
            has_sin = self.sin_dot in diacritics and char in self.can_have_sin_dot
            sin_labels.append(1 if has_sin else 0)
        
        return (
            unvocalized_line,
            vocalized_line,
            torch.tensor(vowel_labels, dtype=torch.long),
            torch.tensor(dagesh_labels, dtype=torch.long),
            torch.tensor(sin_labels, dtype=torch.long),
        )
    
    def _extract_diacritics(self, unvocalized: str, vocalized: str) -> dict:
        """Extract diacritics for each character position"""
        char_to_diacritics = {}
        unvocalized_idx = 0
        
        for i, char in enumerate(vocalized):
            # Skip diacritics
            if self.nikud_pattern.match(char):
                continue
            
            # Found a character
            if unvocalized_idx < len(unvocalized):
                # Collect all diacritics that follow this character
                diacritics = []
                j = i + 1
                while j < len(vocalized) and self.nikud_pattern.match(vocalized[j]):
                    diacritics.append(vocalized[j])
                    j += 1
                
                char_to_diacritics[unvocalized_idx] = diacritics
                unvocalized_idx += 1
        
        return char_to_diacritics


class Collator:
    """Collates individual training examples into batches."""

    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def collate_fn(self, items: List[Tuple]) -> Batch:
        """Collate individual items into a batch."""
        text_list = [item[0] for item in items]
        vocalized_list = [item[1] for item in items]
        vowel_targets_list = [item[2] for item in items]
        dagesh_targets_list = [item[3] for item in items]
        sin_targets_list = [item[4] for item in items]

        # Tokenize all texts in the batch
        tokenized_inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        # Create target tensors matching tokenized sequence length
        batch_size = len(text_list)
        sequence_length = tokenized_inputs.input_ids.size(1)

        vowel_batch_targets = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        dagesh_batch_targets = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        sin_batch_targets = torch.zeros(batch_size, sequence_length, dtype=torch.long)

        # Map character-level targets to token-level targets
        self._map_character_targets_to_tokens(
            vowel_targets_list,
            dagesh_targets_list,
            sin_targets_list,
            tokenized_inputs.offset_mapping,
            vowel_batch_targets,
            dagesh_batch_targets,
            sin_batch_targets,
        )

        # Remove offset mapping from final inputs (not needed for training)
        del tokenized_inputs["offset_mapping"]

        return Batch(
            text=list(text_list),
            vocalized=list(vocalized_list),
            input=tokenized_inputs,
            vowel_targets=vowel_batch_targets,
            dagesh_targets=dagesh_batch_targets,
            sin_targets=sin_batch_targets,
        )

    def _map_character_targets_to_tokens(
        self,
        vowel_targets_list: List[torch.Tensor],
        dagesh_targets_list: List[torch.Tensor],
        sin_targets_list: List[torch.Tensor],
        offset_mappings: torch.Tensor,
        vowel_batch_targets: torch.Tensor,
        dagesh_batch_targets: torch.Tensor,
        sin_batch_targets: torch.Tensor,
    ) -> None:
        """Map character-level targets to token-level targets using offset mappings."""
        for batch_idx, (vowel_targets, dagesh_targets, sin_targets, token_offsets) in enumerate(
            zip(vowel_targets_list, dagesh_targets_list, sin_targets_list, offset_mappings)
        ):
            for token_idx, (char_start, char_end) in enumerate(token_offsets):
                char_start, char_end = int(char_start), int(char_end)
                if char_end > 0 and char_start < len(vowel_targets):
                    # For single-character tokens, use that character's label
                    # For multi-character tokens, use the first character's label
                    vowel_batch_targets[batch_idx, token_idx] = vowel_targets[char_start]
                    dagesh_batch_targets[batch_idx, token_idx] = dagesh_targets[char_start]
                    sin_batch_targets[batch_idx, token_idx] = sin_targets[char_start]


def get_dataloader(
    unvocalized_lines: list[str],
    vocalized_lines: list[str],
    batch_size: int,
    collator: Collator,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """Create a DataLoader for training or validation"""
    return DataLoader(
        TrainData(unvocalized_lines, vocalized_lines),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator.collate_fn,
        num_workers=num_workers,
    )

