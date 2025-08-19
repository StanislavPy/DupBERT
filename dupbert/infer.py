import yaml
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from .model import DupBERT
from .transforms import TextTokenizer, Encoder, PadSequencer


class DupBERTService:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        checkpoint_path: str = "checkpoints/best.pth",
    ):
        with open(config_path, "r") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Build preprocessing stages exactly like in training ---
        # If your config uses different keys, adjust here.
        pre_cfg = self.cfg.get("preprocessing", {})
        self.tokenizer = TextTokenizer(**pre_cfg.get("tokenizer", {}))
        self.encoder = Encoder(**pre_cfg.get("encoder", {}))
        self.pad = PadSequencer(**pre_cfg.get("pad", {}))

        # --- Build model exactly like in training ---
        model_cfg = self.cfg.get("model", {})
        self.model = DupBERT(model_cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state.get("model_state_dict", state))
        self.model.eval()

        self.threshold = float(self.cfg.get("inference", {}).get("threshold", 0.5))

    def _make_batch(self, text_a: str, text_b: str) -> Dict[str, torch.Tensor]:
        """
        Turn raw strings into the tensors your model expects.
        Adjust this method to mirror your training pipeline exactly.
        """
        # 1) Clean/tokenize
        toks_a = self.tokenizer(text_a)
        toks_b = self.tokenizer(text_b)

        # 2) BERT (or your) encoder to get input_ids, masks, etc.
        enc_a = self.encoder(toks_a)
        enc_b = self.encoder(toks_b)

        # 3) Pad/pack to tensors
        batch = self.pad(enc_a, enc_b)  # Design your PadSequencer to return dict of tensors
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        return batch

    @torch.inference_mode()
    def score_pair(self, text_a: str, text_b: str) -> Tuple[float, bool]:
        batch = self._make_batch(text_a, text_b)

        # --- TODO: call your model exactly as trained ---
        # Common pattern for a siamese model (rename keys if different):
        # prob_logit = self.model(
        #     batch["input_ids_a"], batch["attention_mask_a"],
        #     batch["input_ids_b"], batch["attention_mask_b"],
        # )
        prob_logit = self.model(**batch)  # if your model accepts a dict

        # If your model already outputs a probability, remove sigmoid()
        if hasattr(prob_logit, "sigmoid"):
            prob = prob_logit.sigmoid().item()
        else:
            prob = float(prob_logit)

        is_dup = prob >= self.threshold
        return prob, is_dup