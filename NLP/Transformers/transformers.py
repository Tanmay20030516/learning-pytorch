# github repo - https://github.com/bentrevett/pytorch-seq2seq/
# blog - https://jalammar.github.io/illustrated-transformer/
# detailed code - https://nlp.seas.harvard.edu/2018/04/03/attention.html
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        """
        Initialization
        :param num_heads: number of attention heads
        :param embed_dim: embedding size
        """
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.heads_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert self.heads_dim*num_heads == embed_dim, "embed dim must be divisible by num_heads"

        # below set of fc layers help learn different relationships across different heads
        self.queries_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.values_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # this layer projects the new learned context into embedding dimension
        # concatenation of outputs from each head takes place, and these outputs
        # are then projected to correct dims
        self.fc_out = nn.Linear(self.heads_dim*self.num_heads, self.embed_dim)

    def forward(self, queries, keys, values, mask):
        """
        Forward pass for self attention
        :param queries: `(N, d_q, embed_dim)`; `d_q` is the length of query
        :param keys: `(N, d_k, embed_dim)`
        :param values: `(N, d_v, embed_dim)`
        :param mask: a padding mask of shape `(N, num_heads, d_q, d_k)`
                     that masks the `<PAD>` token, so that those `<PAD>`
                     positions do not contribute to softmax
        :return: output contextual vector
        """
        batch_size = queries.shape[0]
        d_q, d_k, d_v = queries.shape[1], keys.shape[1], values.shape[1]

        queries = self.queries_fc(queries)
        keys = self.keys_fc(keys)
        values = self.values_fc(values)

        # split queries, keys and values according to number of heads
        queries = queries.reshape(batch_size, d_q, self.num_heads, self.heads_dim)
        keys = keys.reshape(batch_size, d_k, self.num_heads, self.heads_dim)
        values = values.reshape(batch_size, d_v, self.num_heads, self.heads_dim)

        # now we will move towards self attention
        # softmax(Q.K.T / sqrt(d_k)).V; scaled-dot product attention

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # queries shape: (N, d_q, num_heads, heads_dim)
        # keys shape: (N, d_k, num_heads, heads_dim)
        # energy: (N, num_heads, d_q, d_k)

        energy = energy.masked_fill(mask == 0, float(-1e20))  # fills the position of <PAD> token with -inf

        scaled_dp_attention = torch.softmax((energy / d_k ** 0.5), dim=3)
        # attention shape: (N, num_heads, d_q, d_k)
        # consider a query and  key of a particular batch and head
        # how much your query attends to/is similar to key
        # q1 = 0.3*k1 + 0.5*k2 + 0.2*k3

        out = torch.einsum("nhql, nlhd->nqhd", [scaled_dp_attention, values])
        # scaled_dp_attention shape: (N, num_heads, d_q, d_k)
        # also, d_k == d_v
        # values shape: (N, d_v, num_heads, heads_dim)
        # out shape: (N, d_q, num_heads, heads_dim)
        out = out.reshape(batch_size, d_q, self.num_heads*self.heads_dim)  # flatten the last two dimensions

        out = self.fc_out(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = SelfAttention(num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion*embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_dim, embed_dim),
        )

    def forward(self, queries, keys, values, mask):
        # 1st group of layers
        sdp_attn = self.attention(queries, keys, values, mask)
        x = self.layer_norm1(sdp_attn + queries)
        x = self.dropout(x)
        # 2nd group of layers
        ff = self.ffn(x)
        out = self.layer_norm2(ff + x)
        out = self.dropout(out)

        return out


class PositionalEmbedding(nn.Module):
    """
    Model contains no recurrence and no convolution, so to let the model know regarding the order of sequence
    some positional encoding is to be done for each word relative to it's position in the sequence
    https://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
    """

    def __init__(self, embed_dim, dropout_prob, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.positional_embedding = self.make_pos_emb(max_len, embed_dim)

    @staticmethod
    def make_pos_emb(max_len, embed_dim):
        pos_emb = torch.zeros(max_len, embed_dim)  # shape: (max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)  # shape: (max_len, 1)
        # 10000.0 ** (torch.arange(0, embed_dim, step=2)/embed_dim)
        # torch.exp(torch.log(10000.0 ** (torch.arange(0, embed_dim, step=2)/embed_dim)))
        denominator = torch.exp(
            (torch.arange(0, embed_dim, step=2) / embed_dim) * torch.log(torch.Tensor([10000.0])).item()
        )
        pos_emb[:, 0::2] = torch.sin(position / denominator)  # even indices in embed_dim
        pos_emb[:, 1::2] = torch.cos(position / denominator)  # odd indices in embed_dim
        pos_emb = pos_emb.unsqueeze(0)  # shape: (1, max_len, embed_dim)

        return pos_emb

    def forward(self, x):
        # x.shape: (N, seq_len, embed_dim)
        x = x + self.positional_embedding[:, : x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, num_layers, num_heads,
                 device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_dim)
        # self.position_embedding = nn.Embedding(max_length, embed_dim)  # diff than paper, but still works
        self.position_embedding = PositionalEmbedding(embed_dim, dropout, max_length)  # the one mentioned in paper

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim, dropout, num_heads, forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        # here in encoder, q k v all are same
        for layer in self.layers:
            # according to paper, 6 such blocks were used
            out = layer(out, out, out, mask)

        return out




