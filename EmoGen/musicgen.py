import torch
import numpy as np
import os, sys, optparse
import config as config
import utils as utils
from config import device, model as model_config
from musicgen_model import PerformanceRNN
from sequence import EventSeq, Control, ControlSeq
from PIL import Image


# pylint: disable=E1101,E1102


# ========================================================================
# Settings
# ========================================================================

# ------------------------------------------------------------------------

output_dir = None
sess_path = 'models/sota_musicgen.sess'
max_len = config.train['window_size']
#greedy_ratio = 0.8
greedy_ratio = config.train['greedy_ratio']
control = None
use_beam_search = False
beam_size = 0
temperature = config.train['temperature']
init_zero = True
image_path = None

if use_beam_search:
    greedy_ratio = "DISABLED"
else:
    beam_size = "DISABLED"

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

controls = None
control = "NONE"
assert (
        max_len > 0
), "either max length or control sequence length should be given"

# ------------------------------------------------------------------------

if __name__ == '__main__':
    print("-" * 70)
    print("Session:", sess_path)
    print("Batch size:", batch_size)
    print("Max length:", max_len)
    print("Greedy ratio:", greedy_ratio)
    print("Beam size:", beam_size)
    print("Output directory:", output_dir)
    print("Controls:", control)
    print("Temperature:", temperature)
    print("Init zero:", init_zero)
    print("-" * 70)

# ========================================================================
# Generating
# ========================================================================

state = torch.load(sess_path, map_location=device)
model = PerformanceRNN(**state["model_config"]).to(device)
print(state["model_config"])
model.load_state_dict(state["model_state"])
model.requires_grad_(True)
model.train()
# print(model)
# print("-" * 70)

def forward(feature = None, batch_size = 1, output_type = 'index'):
    
    if init_zero:
        init = torch.zeros(batch_size, model.init_dim).to(device)
    else:
        init = torch.randn(batch_size, model.init_dim).to(device)
    
    feature = feature.reshape(batch_size, model.gru_layers, model.feature_dim).permute(1,0,2)
    
    if output_type=="index":
        model.sampling_ratio = 1

    outputs = model.generate(
        init,
        max_len,
        greedy=greedy_ratio,
        temperature=temperature,
        verbose=False,
        feature=feature,
        output_type = output_type
    )
    model.sampling_ratio = 0
    return outputs

def save(outputs, name):
    output = outputs.cpu().detach().numpy().T[0]  # [batch, steps]
    torch.cuda.empty_cache()
    name = os.path.basename(name).split(".")[0]
    output_dir = "outputs/"
    output_path = os.path.join(output_dir, name) + ".MID"
    n_notes = utils.event_indeces_to_midi_file(output, output_path)
    print(f"===> {output_path} ({n_notes} notes)")
    
if __name__ == '__main__':
    outputs = forward()
    save(outputs, name='vanilla')
