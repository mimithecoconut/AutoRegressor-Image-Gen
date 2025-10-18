import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        decoder_layer = torch.nn.TransformerEncoderLayer(d_latent, nhead=2, dropout=0.1)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        x_seq = x.view(B, -1) # (B, L = H * W)

        x = self.embedding(x_seq) # (B, L, d_latent)
        x = x.transpose(1, 2) # transpose for using ConstPad1d
        shift = torch.nn.ConstantPad1d((1,0),0) # Pad value 0 to the left
        x = shift(x) # (B, L, d+1)
        x = x[:,:,:-1] # Remove last token to maintain length
        x = x.transpose(1, 2)

        x = x.transpose(0, 1) # Since transformer takes in shape (L, B, d)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(h*w).to(x.device)
        x_encoded = self.decoder(x, mask=causal_mask)
        x_encoded = x_encoded.transpose(0, 1)
        logits = self.output_layer(x_encoded) # (B, L, n_tokens)

        return logits.view(B, h, w, self.n_tokens), {}
        

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        L = h * w
        with torch.no_grad():
          x = torch.zeros(B, L, dtype=torch.long, device=device)

          for i in range(L):
              logits, _ = self.forward(x.view(B, h, w))  #(B, h, w, n_tokens)
              logits_flat = logits.view(B, L, self.n_tokens)
              probs = torch.nn.functional.softmax(logits_flat[:, i], dim=-1)
              next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
              x[:, i] = next_token

        return x.view(B, h, w)
