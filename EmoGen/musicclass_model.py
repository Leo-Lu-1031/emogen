import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
from progress.bar import Bar
from config import device
from PIL import Image
import math


# pylint: disable=E1101,E1102

# model = torch.hub.load("pytorch/vision:v0.4.2", "resnet18", pretrained=True)


class PerformanceRNN(nn.Module):
    def __init__(
            self,
            event_dim,
            control_dim,
            init_dim,
            hidden_dim,
            gru_layers=3,
            gru_dropout=0.3,
    ):
        super().__init__()

        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.input_dim = event_dim
        self.output_dim = 4

        gru_dropout = 0
        
        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )

        self.output_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_dim)
        )
            
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for n in self.output_fc: 
            if type(n)==nn.Linear: 
                nn.init.xavier_normal_(n.weight)
                n.bias.data.fill_(0.0)

        # nn.init.xavier_normal_(self.Q.weight)
        # self.Q.bias.data.fill_(0.0)
        # nn.init.xavier_normal_(self.K.weight)
        # self.K.bias.data.fill_(0.0)
        # nn.init.xavier_normal_(self.V.weight)
        # self.V.bias.data.fill_(0.0)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)

    def init_to_hidden(self, init, images):
        # [batch_size, init_dim]
        batch_size = init.shape[0]

        out = self.load_feature_map(images)

        out = self.inithid_fc(out)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out

    def expand_controls(self, controls, steps):
        # [1 or steps, batch_size, control_dim]
        assert len(controls.shape) == 3
        assert controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.repeat(steps, 1, 1)

    def generate(
            self,
            steps,
            events=None,
            controls=None,
            teacher_forcing_ratio=1.0,
            output_type="index",
            verbose=False,
    ):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        batch_size = events.shape[1]
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 3
            assert events.shape[0] >= steps - 1, events.shape
            event = events[0]
            events = events[1:]

        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        #hidden = self.init_to_hidden(init, images)
        
        _, hidden = self.gru(events)
        outputs = self.output_fc(hidden[-1])
    
        if output_type == 'index': return outputs.argmax(dim=-1)
        if output_type == 'softmax': 
            return self.output_fc_activation(outputs)
        if output_type == 'logit': return outputs
