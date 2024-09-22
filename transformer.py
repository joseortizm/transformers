import torch
import torch.nn as nn
import math
import torch.nn.functional as F

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device("cpu")


MAX_SEQ_LEN = 128

class PositionalEmbedding(nn.Module):
    """
    Positional Encoding (Attention Is All You Need): https://arxiv.org/abs/1706.03762

        pos_embed_matrix: rows is max sequence (in sentence), col is d_model (numbre of elements in embedding)
                          default to all zeros

        token_pos: token position
        div_term:  10000^(2i/d_model)
        pos_embed_matrix[:, 0::2]: even
        pos_embed_matrix[:, 1::2]: odd
    """

    def __init__(self, d_model, max_seq_len = MAX_SEQ_LEN):
        super().__init__()
        self.pos_embed_matrix = torch.zeros(max_seq_len, d_model, device=device)

        token_pos = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        self.pos_embed_matrix[:, 0::2] = torch.sin(token_pos * div_term)
        self.pos_embed_matrix[:, 1::2] = torch.cos(token_pos * div_term)
        self.pos_embed_matrix = self.pos_embed_matrix.unsqueeze(0).transpose(0,1)

    def forward(self, x):
        """
        add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks
        """
        return x + self.pos_embed_matrix[:x.size(0), :]
        

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Attention Is All You Need): https://arxiv.org/abs/1706.03762 

    An attention function can be described as mapping a query and a set of key-value pairs to an output, 
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
    of the values, where the weight assigned to each value is computed by a compatibility function of 
    the query with the corresponding key.

    MultiHead(Q, K, V) = Concat(head1, head2, ...headh)* W_o
    Where headi = Attention(W_q(Q) , W_k(K) , W_v(V)) 

    (Linear Projection of Queries, Keys, and Values:)
    W_q(Q): projected queries for head i
    W_k(K): Projected keys for head i
    W_v(V): Projected values for head i

    In this work we employ h = 8 parallel attention layers, or heads. 
    For each of these we use dk = dv = dmodel/h = 64. 
    
    Q: Queries
    K: Keys
    V: Values
    Q, K and V: [batch_size, seq_len, d_model]

    d_v, d_k: dimension of each head

    W_q, W_k, W_v, W_o: - weights
                        - dimensions: if d_model = 512 and num_heads = 8 so dimensions for head is 512/8 = 64
                            -> 8*(512,64) connections is = 512,512
    """ 
    def __init__(self, d_model = 512, num_heads = 8):
        super().__init__()
        assert d_model % num_heads == 0, 'Embedding size not compatible with num heads'

        self.d_v = d_model // num_heads
        self.d_k = self.d_v  #d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask = None):
        """
        batch_size: examples for batch

        Q, K and V: organizationi and data preparation for each head ->[batch_size, num_heads, seq_len, d_k]
        
        Attention: the relevance or importance of each token in the input sequence relative to other tokens
                   the tensor that contains the calculated attention probabilities.
        
        weighted_values: is the final result of the attention mechanism after applying the attention scores to the values (V)
                         the tensor that contains the weighted values calculated using the attention probabilities.
                          = attention * V

        Note:
            weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
            
            1)This operation is essential for the concatenation of the outputs from all the attention heads, 
            allowing the model to effectively combine and utilize the information extracted by each head
            before the final projection:
                Transposition: Reorganizes the dimensions to facilitate concatenation.
                Contiguity: Ensures that the tensor is contiguous in memory, allowing for efficient manipulation.
                Reshaping: Combines the outputs of the attention heads into a single feature dimension.

            2)is indeed used to revert the changes made during the transformation of 
            the input tensors (queries, keys, and values) in the earlier part of the code:

            Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        """
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) 

        weighted_values, attention = self.scale_dot_product(Q, K, V, mask)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
        weighted_values = self.W_o(weighted_values)
        
        return weighted_values, attention

    def scale_dot_product(self, Q, K, V, mask = None):
        """
        Attention(Q, K, V) = softmax( (Q*K^T) / (dk)^(1/2) ) * V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim = -1)
        weighted_values = torch.matmul(attention, V)
        return weighted_values, attention

class PositionFeedForward(nn.Module):
    """
     Position-wise Feed-Forward Networks (Attention Is All You Need): https://arxiv.org/abs/1706.03762

    In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected 
    feed-forward network, which is applied to each position separately and identically. This consists of 
    two linear transformations with a ReLU activation in between.

    FFN(x) = max(0, xW1 + b1)W2 + b2   

    While the linear transformations are the same across different positions, they use different parameters from 
    layer to layer.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderSubLayer(nn.Module):
    """
    Residual Dropout: We apply dropout to the output of each sub-layer, 
                    before it is added to the sub-layer input and normalized. In addition, we apply dropout 
                    to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. 
                    For the base model, we use a rate of Pdrop = 0.1.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.droupout1 = nn.Dropout(dropout)
        self.droupout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Multi-Head Attention is once in Encoder and twice in Decoder so you can use one or two Multi-Head Attetion 
        (Encoder and Decoder). 
        In this code one Multi-Head Attention class so:
            Multi-Head Attention needs Q, K and V so for this implementation three X will be used in self.self_attn().
        """
        attention_score, _ = self.self_attn(x, x, x, mask)
        x = x + self.droupout1(attention_score)
        x = self.norm1(x)
        x = x + self.droupout2(self.ffn(x))
        return self.norm2(x)


class Encoder(nn.Module):
    """
    layers: Encoder layers (Nx = 6)
    norm: Layer Normalization (optional: If you want to further stabilize the final output of the Encoder, you can keep the final normalization)
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderSubLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderSubLayer(nn.Module):
    """
    attention_score: first sub-layer
    cross_attn (Cross-Attention): second sub-layer
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, target_mask=None, encoder_mask=None):
        attention_score, _ = self.self_attn(x, x, x, target_mask)
        x = x + self.dropout1(attention_score)
        x = self.norm1(x) #go down to self.cross_attn()
        
        encoder_attn, _ = self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        x = x + self.dropout2(encoder_attn)
        x = self.norm2(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        return self.norm3(x)



class Decoder(nn.Module):
    """
    layers: Decoder layers (Nx = 6)
    norm: Layer Normalization (optional: If you want to further stabilize the final output of the Decoder, you can keep the final normalization)
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderSubLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, target_mask, encoder_mask):  
        """
        x: "outputs(shifted right)" 
        encoder_output: output of Encoder
        
        Mask: We also modify the self-attention sub-layer in the decoder stack to prevent positions 
        from attending to subsequent positions. This masking, combined with fact that the output 
        embeddings are offset by one position, ensures that the predictions for position i can 
        depend only on the known outputs at positions less than i.
        
        target_mask (Masked Multi-Head Attention), encoder_mask (mask for encoder. e.g. don't pay attention to paddings)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, encoder_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
                To facilitate these residual connections, all sub-layers in the model, 
                as well as the embedding layers, produce outputs of dimension dmodel = 512.
   
        num_heads: the number of heads in the multiheadattention models (default=8).
                  In this work we employ h = 8 parallel attention layers, or heads.
    
        d_ff (dim_feedforward): the dimension of the feedforward network model (default=2048).
                                The inner-layer has dimensionality = 2048.
    
        num_layers: The encoder is composed of a stack of N = 6 identical layers. 
                The decoder is also composed of a stack of N = 6 identical layers.
                in Pytorch (https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html):
                    num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
                    num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).

        input_vocab_sizes: size of dictionary (english origin language)

        target_vocab_size: size of dictionary

        max_len: max number of tokens in a statement

        dropout: For the base model, we use a rate of Pdrop = 0.1
    -
    """

    def __init__(self, d_model, num_heads, d_ff, num_layers, input_vocab_size, 
    target_vocab_size, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model) # Input Embedding
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model) # Output Embedding
        self.pos_embedding = PositionalEmbedding(d_model, max_len) # Positional Encoding (paper)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, target_vocab_size) # d_model: number of elements passed in the layers 

    
    def forward(self, source, target):
        """
        Input Embedding:  In the embedding layers, we multiply those weights by square root of d_model
                        >>> self.encoder_embedding(source) * math.sqrt(self.encoder_embedding.embedding_dim)
                        where self.encoder_embedding.embedding_dim is d_model: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#embedding

        Positional Encoding: class PositionalEmbedding()

        same for target
        """
        source_mask, target_mask = self.mask(source, target)

        source = self.encoder_embedding(source) * math.sqrt(self.encoder_embedding.embedding_dim)
        source = self.pos_embedding(source)
        encoder_output = self.encoder(source, source_mask) #Create Encoder and get output 

        target = self.decoder_embedding(target) * math.sqrt(self.decoder_embedding.embedding_dim)
        target = self.pos_embedding(target)
        output = self.decoder(target, encoder_output, target_mask, source_mask) #Create Decoder. source_mask is encoder_mask 
        # use of softmax will be performed during training
        return self.output_layer(output)


    def mask(self, source, target):
        """
        Generate mask for Transformer
        """
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        size = target.size(1) #represents the tokens in the sequence
        no_mask = torch.tril(torch.ones((1, size, size), device=device)).bool() #elements on and below the main diagonal are True, and elements above are False.
        target_mask = target_mask & no_mask #True only at positions where both (target_mask and no_mask) are True.
        return source_mask, target_mask