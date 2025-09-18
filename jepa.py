import math
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from models import load_pretrained_encoder_weights


class JEPAEncoder(nn.Module):
    def __init__(
        self, model_name: str = "t5-base", encoder_state_path: str | None = None
    ):
        super().__init__()
        # load nguyên model T5 (có cả encoder + decoder)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = self.t5.encoder  # lấy encoder

        # Nếu có file weight pretrain encoder riêng thì load vào
        if encoder_state_path is not None:
            missing_keys, unexpected_keys = load_pretrained_encoder_weights(
                self.t5, encoder_state_path
            )
            if missing_keys or unexpected_keys:
                print(
                    f"[Warning] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
                )

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: tensor [batch_size, seq_len]
        attention_mask: tensor [batch_size, seq_len] (optional)
        return_hidden_state: nếu True thì trả hidden_state cuối cùng
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        return encoder_outputs.last_hidden_state


class JEPAPredictor(nn.Module):
    def __init__(
        self, model_name: str = "t5-base", encoder_state_path: str | None = None
    ):
        super().__init__()
        # load nguyên model T5 (có cả encoder + decoder)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = self.t5.encoder  # lấy encoder

        # Nếu có file weight pretrain encoder riêng thì load vào
        if encoder_state_path is not None:
            missing_keys, unexpected_keys = load_pretrained_encoder_weights(
                self.t5, encoder_state_path
            )
            if missing_keys or unexpected_keys:
                print(
                    f"[Warning] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
                )

    def forward(self, inputs_embeds, attention_mask=None):
        """
        inputs_embeds: tensor [batch_size, seq_len, hidden_size]
        attention_mask: tensor [batch_size, seq_len] (optional)
        return: last hidden state from encoder
        """
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,  # Truyền trực tiếp hidden state
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state


def init_sinusoidal_embeddings(embedding_layer: nn.Embedding):
    """
    Khởi tạo weights của nn.Embedding bằng giá trị sinusoidal.
    - embedding_layer: Layer nn.Embedding cần init.
    """
    num_positions = embedding_layer.num_embeddings
    hidden_size = embedding_layer.embedding_dim
    position = torch.arange(0, num_positions).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
    )
    pe = torch.zeros(num_positions, hidden_size)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    embedding_layer.weight.data = pe
