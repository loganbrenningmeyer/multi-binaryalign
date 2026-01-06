import torch
import torch.nn as nn
from transformers import AutoModel

from binaryalign.models.classifier import BinaryAlignClassifier


class BinaryAlignModel(nn.Module):
    def __init__(self, backbone: AutoModel, classifier: BinaryAlignClassifier):
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        # ----------
        # Get backbone final hidden state
        # ----------
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)

        # ----------
        # Classify target alignment / mask for target tokens
        # ----------
        logits = self.classifier(hidden_states)  # (B, L, 1)
        logits = logits.squeeze(-1)  # (B, L)

        logits_target = logits[target_mask]  # (T)

        return logits_target
    
    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        threshold: float=0.5
    ):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        self.eval()

        logits = self.forward(input_ids, attention_mask, target_mask)
        scores = torch.sigmoid(logits)
        preds = scores >= threshold

        return preds
