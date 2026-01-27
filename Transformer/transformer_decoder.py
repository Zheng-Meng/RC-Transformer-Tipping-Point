# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 50000)
        transformer_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.init_weights()
        self.d_model = d_model
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        device = src.device
        causal_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)
        
        output = self.transformer_encoder(src, mask=causal_mask)
        output = self.decoder(output)
        
        return output  # Return full sequence prediction now

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int) -> None:
        super().__init__()

        p_pos = torch.arange(0, seq_length).unsqueeze(1).to(device)
        p_i = torch.arange(0, d_model).to(device)

        PE = (p_pos / (1000**(2*p_i/d_model))).unsqueeze(0)
        PE[0, :, 0::2] = torch.sin(PE[:, :, 0::2])
        PE[0, :, 1::2] = torch.cos(PE[:, :, 1::2])
        self.PE = PE.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.PE[:, :x.shape[1], :]
