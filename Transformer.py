import os
import sys
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.emb_dim = emb_dim

        if emb_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        
        self.head_dim = emb_dim // num_heads
        self.query_layer =  nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        
        self.output_layer = nn.Linear(num_heads * self.head_dim, emb_dim)

    def split_heads(self, tensor, batch_size):
        # Split tensor into (batch_size, num_heads, seq_length, head_dim)
        seq_len = tensor.size(1)  # Get the sequence length from the tensor
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query_val = self.query_layer(query)
        key_val = self.key_layer(key)
        value_val = self.value_layer(value)

        query_val = self.split_heads(query_val, batch_size)
        key_val = self.split_heads(key_val, batch_size)
        value_val = self.split_heads(value_val, batch_size)

        key_out = torch.matmul(query_val, key_val.permute(0, 1, 3, 2)) 

        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to ignore masked positions
        key_out = F.softmax(key_out / math.sqrt(self.head_dim), dim=-1)
        key_out = key_out * mask
        # Weighted sum of values
        out = torch.matmul(key_out, value_val)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.emb_dim)

        # Final linear transformation
        out = self.output_layer(out)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, forward_dim*emb_dim),
            nn.ReLU(),
            nn.Linear(forward_dim*emb_dim, emb_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):
        
        attention_out = self.attention(query, key, value, mask)
        attention_out = self.dropout(attention_out)
        attention_out = self.norm1(attention_out + query)
        
        ffn_out = self.ff(attention_out)
        ffn_out = self.dropout(ffn_out + attention_out)
        ffn_out = self.norm1(ffn_out)
        
        

        return ffn_out
    
def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        
        # Generate the sinusoidal positional encodings
        positional_encoding = get_sinusoid_table(max_len, emb_dim)
        positional_encoding = positional_encoding.clone().detach() # torch.tensor(positional_encoding, dtype=torch.float32)
        
        # Create an nn.Embedding using .from_pretrained
        self.embedding = nn.Embedding.from_pretrained(
            positional_encoding, freeze=True  # Freeze ensures embeddings are not trainable
        )

    def forward(self, positions):
        # Fetch the positional encodings corresponding to the input positions
        return self.embedding(positions)

# Encoder class 
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim 
        self.num_layers = num_layers

        # getting input sentence embedding 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # getting positional embedding 
        self.positional_encoding = PositionalEncoding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        # creating list of encoder layers 
        self.layers= nn.ModuleList([TransformerBlock(emb_dim, num_heads, dropout, forward_dim) 
                                    for _ in range(num_layers)])

    def forward(self, x, mask):
    
        # n = number of examples and seq lenth of input
        n, seq_len = x.size() 
        #pos =torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # moving positions to device 
        # on which input (x) is located 
        pos = torch.arange(1, seq_len+1, device = x.device).expand(n, seq_len)
        
        
        # we then run our input through the embeddings.
        embedding = self.embedding(x)
        
        # The created positions are run through the positional encoding layer
        positional_encodings = self.positional_encoding(pos)
        
        #The results are summed up
        x = embedding + positional_encodings
        
        # Apply dropout
        x = self.dropout(x)
        
        # The output is then passed through the transformer blocks where x is the query, key, and value
        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x 
    
    
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.attention = MultiHeadAttention(emb_dim, num_heads) 
        
        self.transformer_block = TransformerBlock(  
            emb_dim, num_heads, dropout, forward_dim
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        self_attention = self.attention(x, x, x, tgt_mask)
        
        query = self.dropout(self.norm(self_attention + x))
        
        x = self.transformer_block(query, key, value, src_mask)
        
        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim) 
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList( # Changed to ModuleList from sequential
            [ #I deleted the '*' from here, was it needed specifically for sequential?
            DecoderBlock(
                emb_dim,
                num_heads,
                forward_dim,
                dropout)
            for _ in range(num_layers)])
        
        self.relative_positional_encoding = nn.Embedding(max_len, emb_dim)
        
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #Through normal embeddings
        batch_size, seq_len = x.size()
        word_embed = self.word_embedding(x) 
        
        # Seems corret as it creates rows of indices from 0 to seq_len
        position_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embed = self.relative_positional_encoding(position_indices) 
        
        #Adding the positional embeddings to the word embeddings and applying dropout
        x = word_embed + pos_embed
        x = self.dropout(x)
        
        
        # I added a second "encoder_output" argument to the forward method
        # as the decoder block needs  x, value, key, src_mask, tgt_mask as arguments
        # and the encoder_output is the value and key??
        # What else could the input be?
        
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                encoder_output,
                src_mask,
                tgt_mask)
            
        
        
        
        x = self.fc(x)  # Shape: (batch_size, seq_len, vocab_size)
        
        return x
    
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        
        # Initialize Encoder and Decoder
        self.encoder = Encoder(src_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        
        # Final Linear Layer for output projection
        self.output_layer = nn.Linear(emb_dim, tgt_vocab_size)
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    def forward(self, src, tgt): 
        
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        # pass through encoder
        encoder_output = self.encoder(src, src_mask)
        
        # pass through decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_layer(decoder_output)
        
        return output