from torch import nn
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.util import CoordBatchConverter 


class CoordsEncoder(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.alphabet = alphabet

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        return_all_hiddens: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        
        return encoder_out
    
    def encode(self, coords, confidence=None, device=None):
        """
        Args:
            coords: L x 3 x 3 list representing one backbone
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.alphabet)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, None, None)], device=device)
        )
        encoder_out = self.forward(batch_coords, padding_mask, confidence)
        # remove beginning and end (bos and eos tokens)
        return encoder_out['encoder_out'][0][1:-1, 0]
