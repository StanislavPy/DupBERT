from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import BertModel

from .state_keys import StateKeys
from .config_reader import ConfigReader


class DupBERT(torch.nn.Module):
    """Duplicates detector model with pretrained BERT vectors.

    Args:
        pretrained_model_name_or_path (Union[str, Path]): the pretrained public/private
            BERT model (e.g. 'bert-base-uncased')
        dropout_rate (float): dropout rate used throughout the network 
        output_channels (List[int]): the list with the number of channels 
            used in each convolution layer
        kernel_sizes (List[int]): the number of kernel sizes used in each convolution 
            layer. Must have the same length as output_channels. 
    """

    def __init__(self,
                 pretrained_model_name_or_path: Union[str, Path],
                 dropout_rate: float = 0.5,
                 output_channels: List[int] = [3, 4, 5],
                 kernel_sizes: List[int] = [100, 100, 100]):
        
        super().__init__()

        if len(output_channels) != len(kernel_sizes):
            raise ValueError(f"The number of output channels must be equal the number of kernel sizes.")

        self.keys= StateKeys()
        self.bert_encoder = BertModel.from_pretrained(pretrained_model_name_or_path)

        # Freeze BERT layer
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels=self.bert_encoder.config.hidden_size,
                out_channels=output_channels[ind_layer],
                kernel_size=kernel_size)
            for ind_layer, kernel_size in enumerate(kernel_sizes)
        ])
        
        input_fc_size = sum(output_channels)
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(input_fc_size * 2, input_fc_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(input_fc_size, input_fc_size // 2),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(input_fc_size // 2, 1),
            torch.nn.Sigmoid()
        )
        
        # Use GPU if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, checkpoint_filepath: Union[str, Path], config_filepath: Union[str, Path]) -> 'BertDupDetector':
        """Class method for loading previously trained model for prediction.

        Args:
            checkpoint_filepath (Union[str, Path]): the path to the checkpoint (e.g. model.best.pth)
            config_filepath (Union[str, Path]): the path to the config (e.g. config.yaml)

        Returns:
            BertDupDetector: Returns the model with loaded weights in evaluation mode.
        """

        config_reader = ConfigReader.load(config_filepath)
        model = cls(**config_reader.model_params)
        checkpoint_dict = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint_dict)

        return model.eval()

    def forward(self, input_tuple):
        left_sample = input_tuple[self.keys.left_sample]
        right_sample = input_tuple[self.keys.right_sample]

        l_bert_h_s, _ = self.bert_encoder(
            left_sample["input_ids"],
            attention_mask=left_sample["attention_mask"],
            return_dict=False
        )

        r_bert_h_s, _ = self.bert_encoder(
            right_sample["input_ids"],
            attention_mask=right_sample["attention_mask"],
            return_dict=False
        )
        
        l_conv_list = [F.relu(conv1d(l_bert_h_s.permute(0, 2, 1))) for conv1d in self.conv_layers]
        r_conv_list = [F.relu(conv1d(r_bert_h_s.permute(0, 2, 1))) for conv1d in self.conv_layers]

        l_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.size(2)) for x_conv in l_conv_list]
        r_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.size(2)) for x_conv in r_conv_list]

        l_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in l_pool_list], dim=1)
        r_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in r_pool_list], dim=1)
        
        concat_fc = torch.cat([l_fc, r_fc], dim=1)
        logits = self.fc_layers(concat_fc)
        return logits.to(torch.float64)
        