import re
from pathlib import Path
from itertools import chain
from typing import List, Union, Dict

import numpy as np
from razdel import tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer
from string import digits, whitespace, punctuation, ascii_letters

from . import utils


class BaseTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TextTokenizer(BaseTransform):
    """Transforms text into tokens.

    Args:
        lower (bool): Lowers all letters in the texts.
        remove_tags (bool): Removes html tags from texts.
    remove_digits (bool): Removes all digits from texts.
    remove_stop_words (bool): Removes predefined stopwords from text.
    stop_words_lang (str): The NLTK stopwords language.
    remove_punctuation (bool): Removes all punctuation from texts.
    keep_only_known_symbols (bool): Keeps only known symbols like digits,
         punctuation, whitespaces, russian and english letters.
    """
    
    def __init__(self,
                 lower: bool = False,
                 remove_tags: bool = False,
                 remove_digits: bool = False,
                 remove_stop_words: bool = False,
                 stop_words_lang: str = 'english',
                 remove_punctuation: bool = False,
                 keep_only_known_symbols: bool = False):

        self.lower = lower
        self.remove_tags = remove_tags
        self.remove_digits = remove_digits
        self.remove_stop_words = remove_stop_words
        self.keep_only_known_symbols = keep_only_known_symbols
        self.remove_punctuations = remove_punctuation

        self.regex_tags = re.compile(r'<[^<]+?>')
        self.regex_backup_split = re.compile(r'(\W|_)')
        self.regex_camel_case = re.compile(
            '.+?(?:(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])|'
            '(?<=[A-ZА-ЯЁ])(?=[A-ZА-ЯЁ][a-zа-яё])|$)'
        )
        self.stopwords_ru = stopwords.words(stop_words_lang)

        self.known_symbols = set(
            digits + whitespace + punctuation + ascii_letters +
            'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНн'
            'оПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'
        )

        # Use previous regex
        self.dirt_regexp = re.compile('(\n[^\S\n]*.[^\S\n]*)+\n')
        self.repeating_new_lines_regexp = re.compile('\n{3,}')
        self.repeating_spaces_regexp = re.compile('[^\S\n]{3,}')
        self.tags_regexp = re.compile('\[(image|bookmark): [^\]\[]*\]')
        self.repeating_punctuation_regexp = re.compile(
            '(([^\w\s]|_)([^\w]|_)+([^\w\s]|_))'
        )

    def check_known_symbols(self, text: str) -> str:
        clean_symbols = ''
        for symbol in text:
            if symbol not in self.known_symbols:
                clean_symbols += ' '
                continue
            clean_symbols += symbol
        return clean_symbols

    def clean_text_from_tags(self, text: str) -> str:
        """
        Removes all unnecessary symbols from texts (e.g. html tags)
        :param text: text with unnecessary symbols
        :return: clean texts
        """
        return self.regex_tags.sub(' ', text)

    @staticmethod
    def get_tokens(text: str) -> List[str]:
        """
        Finds tokens in the provided text
        :param text: text to process
        :return: list of tokens
        """
        return [x.text for x in tokenize(text)]

    def backup_tokenize(self, tokens: List[str]) -> List[str]:
        """
        Additional tokenizer which checks symbols that
        were not separated in the base tokenizer
        :param tokens: list of tokens
        :return: list of new tokens
        """
        backup_split_tokens = chain.from_iterable(
            [self.regex_backup_split.split(x) for x in tokens]
        )
        camel_case_split_tokens = chain.from_iterable(
            [self.regex_camel_case.findall(x) for x in backup_split_tokens]
        )
        return list(filter(None, camel_case_split_tokens))

    def __call__(self, text: str) -> List[str]:
        """
        Transforms text string document into the list of tokens
        :param text: text string
        :return: list of tokens
        """

        # Use previous regex
        text = self.dirt_regexp.sub('\n', text)
        text = self.repeating_new_lines_regexp.sub('\n\n', text)
        text = self.repeating_spaces_regexp.sub('  ', text)
        text = self.tags_regexp.sub('', text)
        text = self.repeating_punctuation_regexp.sub('\\2\\4', text)

        # Tokenize and join values for more qualitative analysis
        text = ' '.join(self.get_tokens(text))

        if self.keep_only_known_symbols:
            text = self.check_known_symbols(text)
        if self.lower:
            text = text.lower()
        if self.remove_tags:
            text = self.clean_text_from_tags(text)

        text_tokens = text.split()

        if self.remove_stop_words:
            text_tokens = [x for x in text_tokens if x not in self.stopwords_ru]
        if self.remove_punctuations:
            text_tokens = [token.strip(punctuation) for token in text_tokens]
        if self.remove_digits:
            text_tokens = [token.strip(digits) for token in text_tokens]

        text_tokens = self.backup_tokenize(text_tokens)

        return text_tokens


class Encoder(BaseTransform):
    """Encodes the preprocessed text to BPE

    Args:
        pretrained_model_name_or_path (Union[str, Path]): the pretrained public/private
            BERT model (e.g. 'bert-base-uncased')
        add_special_tokens (bool): add special BPE tokens
    """

    def __init__(self,
                 pretrained_model_name_or_path: Union[str, Path] = None,
                 add_special_tokens: bool = False):
        super().__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.add_special_tokens = add_special_tokens

    def get_encoding(self, word: str):
        return self.bert_tokenizer.encode(word, add_special_tokens=self.add_special_tokens)

    def __call__(self, tokenized_text: str) -> List[int]:
        encoded_tokens = list(map(self.get_encoding, tokenized_text))
        return utils.flatten(encoded_tokens)


class PadSequencer(BaseTransform):
    """Padding of the BPE sequences.

    Args:
        max_seq_length (int): the maximum length of the sequence
        d_type (bool): the dtype of the outputing sequences
        padding (bool): pad either before or after each sequence. ['pre', 'post']
        truncating (bool): remove values from sequences larger than
            `max_seq_length`, either at the beginning or at the end of the sequences. 
            ['pre', 'post']
    """

    def __init__(self,
                 max_seq_length: int = 512,
                 d_type: str = 'int64',
                 padding: str = 'post',
                 truncating: str = 'post'):

        super().__init__()

        assert max_seq_length <= 512, 'Max seq length must be lesser than 512'
        self.max_seq_length = max_seq_length
        self.d_type = d_type
        self.padding = padding
        self.truncating = truncating

    def pad_sequence(self, sequence):
        padded_sequence = utils.pad_sequences(
            [sequence], maxlen=self.max_seq_length,
            padding=self.padding, truncating=self.truncating,
            dtype=self.d_type
        ).flatten()

        return padded_sequence

    @staticmethod
    def get_attention_mask(word_indices: np.array) -> np.array:
        mask_indices = word_indices.copy()
        mask_indices[mask_indices > 0] = 1
        return mask_indices

    def __call__(self, bpe_list: List[int]) -> Dict[str, np.array]:
        input_ids = self.pad_sequence(bpe_list)
        out_dict = {
            "input_ids": input_ids,
            "attention_mask": self.get_attention_mask(input_ids)
        }

        return out_dict
