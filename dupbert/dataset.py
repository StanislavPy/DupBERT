from typing import List, Tuple, Dict

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .state_keys import StateKeys
from .transforms import TextTokenizer, Encoder, PadSequencer

TRIPLET = Tuple[str, str, int]


class TripletDataset(Dataset, StateKeys):
    """
    Dataset for duplicates detection task.

    Args:
        triplets: a list of triplets in the format (text, text, class)
        train_mode: mode whether to use torch.no_grad() or not
        txt_tokenizer : tokenized given text
        encoder: changing the bio data format to the BPE format
        pad_sequencer: padding of BPE format sequences to the same length

    Output:
        ({'input_ids': array, 'attention_mask': array}, int)
    """

    def __init__(
            self,
            triplets: List[TRIPLET],
            txt_tokenizer: TextTokenizer,
            encoder: Encoder,
            pad_sequencer: PadSequencer,
            train_mode: bool = True,
    ):
        self.train_mode = train_mode
        self.transforms = Compose([txt_tokenizer, encoder, pad_sequencer])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = list()
        self.prepare_dataset(triplets)

    @staticmethod
    def wrap_sample_dict(sample_dict: Dict[str, np.array]) -> Dict[str, torch.Tensor]:
        wrapped_dict = {}

        for key in sample_dict:
            wrapped_dict[key] = torch.from_numpy(sample_dict[key]).to(self.device)

        return wrapped_dict

    def preprocess_text(self, text: str):
        return self.transforms(text)

    def __add_triplet(self, txt_0: str, txt_1: str, target: int):
        self.data.append(
            {
                self.input_tuple: {
                    self.left_sample: self.preprocess_text(txt_0),
                    self.right_sample: self.preprocess_text(txt_1)
                },
                self.targets: torch.from_numpy(np.array([target])).to(self.device)
            }
        )

    def prepare_dataset(self, triplets: List[TRIPLET]):

        for txt_0, txt_1, target in triplets:
            if self.train_mode:
                self.__add_triplet(txt_0, txt_1, target)
                continue

            with torch.no_grad():
                self.__add_triplet(txt_0, txt_1, target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
