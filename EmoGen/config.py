import torch
from sequence import EventSeq, ControlSeq

# pylint: disable=E1101

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')

model = {
    'init_dim': 512,
    'event_dim': EventSeq.dim(),
    'control_dim': ControlSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0,
    'use_attention': True,
    'sampling_ratio': 0,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 256,
    'window_size': 500,
    'stride_size': 10,
    'use_transposition': False,
    'control_ratio': 0,
    'teacher_forcing_ratio': 1,
    'vae_beta': 0,
    'greedy_ratio': 0.8,
    'temperature': 1.0,
}

def reg(x):
    x.retain_grad()
    x.register_hook(lambda p: print(p.grad))