import torch
import torch.nn as nn

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device("cpu")


MAX_SEQ_LEN = 30 #128

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = MAX_SEQ_LEN):
        pass

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        pass

    def forward(self, x, mask=None):
        pass

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        pass

    def forward(self, x, encoder_output, target_mask=None, encoder_mask=None):  
        """
        -
        x: "outputs(shifted right)" (parcial)
        
        - 
        encoder_output: output of Encoder

        
        Mask: We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
        -
        target_mask (Masked Multi-Head Attention), encoder_mask (mask for encoder. e.g. don't pay attention to paddings)


        """
        pass


class Transformer(nn.Module):
    """
   
    xxx: pytorch name and descrip pytorch
    xxx: video name and descrip paper 
    
    d_model: the number of expected features in the encoder/decoder inputs (default=512).
    d_model: To facilitate these residual connections, all sub-layers in the model, 
    as well as the embedding layers, produce outputs of dimension dmodel = 512.
   
    nhead: the number of heads in the multiheadattention models (default=8).
    num_heads: In this work we employ h = 8 parallel attention layers, or heads.
    

    dim_feedforward: the dimension of the feedforward network model (default=2048).
    d_ff: ... and the inner-layer has dimensionality dff = 2048.

    num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
    num_layers: The encoder is composed of a stack of N = 6 identical layers. 
                The decoder is also composed of a stack of N = 6 identical layers.

    input_vocab_sizes: size of dictionary (english origin language)
    target_vocab_size: size of dictionary
                       output_layer = nn.Linear(...)->probability distribution for each word of the target language

    -
    max_len: max number of tokens in a statement

    -
    dropout: For the base model, we use a rate of Pdrop = 0.1

    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, input_vocab_size, 
    target_vocab_size, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, max_len) #Positional Encoding (paper)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, target_vocab_size) #d_model: number of elements passed in the layers 

    
    def forward(self, source, target):
        pass

    def mask(self, source, target):
        """
        generate mask
        
        """
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)

