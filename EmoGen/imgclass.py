import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import os

try:
    import pretty_midi
except:
    os.system("pip install pretty_midi progress")
    
import sys
import time
import optparse
import utils as utils
import config as config
from dataloader import Dataset
from imgclass_model import ResNet as PerformanceRNN


# pylint: disable=E1102
# pylint: disable=E1101

# ========================================================================
# Settings
# ========================================================================

# ------------------------------------------------------------------------

sess_path = 'models/sota_imgclass.sess'
data_path = "../data/emotion_dataset"
saving_interval = 600

learning_rate = config.train['learning_rate']
batch_size = config.train['batch_size']
window_size = config.train['window_size']
stride_size = config.train['stride_size']
use_transposition = config.train["use_transposition"]
control_ratio = config.train['control_ratio']
teacher_forcing_ratio = config.train['teacher_forcing_ratio']
reset_optimizer = False
enable_logging = False

model_config = config.model
model_params = utils.params2dict('')
model_config.update(model_params)
device = config.device

if __name__ == '__main__':
    print("-" * 70)

    print("Session path:", sess_path)
    print("Dataset path:", data_path)
    print("Saving interval:", saving_interval)
    print("-" * 70)

    print("Hyperparameters:", utils.dict2params(model_config))
    print("Learning rate:", learning_rate)
    print("Batch size:", batch_size)
    print("Window size:", window_size)
    print("Stride size:", stride_size)
    print("Control ratio:", control_ratio)
    print("Teacher forcing ratio:", teacher_forcing_ratio)
    print("Random transposition:", use_transposition)
    print("Reset optimizer:", reset_optimizer)
    print("Enabling logging:", enable_logging)
    print("Device:", device)
    print("-" * 70)


# ========================================================================
# Load session and dataset
# ========================================================================

# load weight and state
def load_session():
    global sess_path, model_config, device, learning_rate, reset_optimizer
    try:
        sess = torch.load(sess_path, map_location=config.device)
        if "model_config" in sess and sess["model_config"] != model_config:
            model_config = sess["model_config"]
            # print("Use session config instead:")
            # print(utils.dict2params(model_config))
        model_state = sess["model_state"]
        optimizer_state = sess["model_optimizer_state"]
        # print("Session is loaded from", sess_path)
        sess_loaded = True
    except:
        print("New session")
        sess_loaded = False
    model = PerformanceRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if sess_loaded:
        model.load_state_dict(model_state)
        if not reset_optimizer:
            optimizer.load_state_dict(optimizer_state)
    assert sess_loaded
    print(model_config)
    return model, optimizer


def load_dataset():
    global data_path
    dataset = Dataset(data_path, verbose=True)
    dataset_size = len(dataset.image_samples)
    assert dataset_size > 0
    return dataset

# print("Loading session")
model, optimizer = load_session()
model.requires_grad_(False)
# print(model)

# print("-" * 70)

# print("Loading dataset")
if __name__ == '__main__':
    dataset = load_dataset()
# print(dataset)

# print("-" * 70)


# ------------------------------------------------------------------------


def save_model(iteration):
    global model, optimizer, model_config, sess_path
    model_name = f""
    print("Saving to", sess_path)
    torch.save(
        {
            "model_config": model_config,
            "model_state": model.state_dict(),
            "model_optimizer_state": optimizer.state_dict(),
        },
        'iteration_{}.sess'.format(iteration),
    )
    print("Done saving")


# ========================================================================
# Training
# ========================================================================

def forward(img, output_type = 'logit'):
    with torch.no_grad():
        outputs = model.forward(img)
        if output_type=='index': return F.one_hot(outputs, 4)
        if output_type=='logit': return outputs
        if output_type=='softmax': return model.output_fc_activation(outputs)
        assert False, 'Invalid output type'

if __name__=='__main__':
    if enable_logging:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
        
    last_saving_time = time.time()
    loss_function = nn.CrossEntropyLoss()
    summer = open("log.txt", "w")
    
    try:
        for iteration in range(10000):
            batch_gen = dataset.paired_batches(batch_size, window_size, stride_size, test = True)
            image_batch, label_batch = batch_gen

            outputs = model.forward(
                images=image_batch,
            )
            loss = loss_function(outputs, F.one_hot(label_batch).float())
    #         model.zero_grad()
    #         loss.backward()

    #         optimizer.step()

            if enable_logging:
                writer.add_scalar("model/loss", loss.item(), iteration)
                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                writer.add_scalar("model/norm", norm.item(), iteration)

            print(f"iter {iteration}, loss: {torch.exp(-loss)}")
            summer.write(f"iter {iteration}, loss: {loss.item()}\n")

            if time.time() - last_saving_time > saving_interval:
                save_model(iteration)
                last_saving_time = time.time()

            del image_batch, label_batch

    except KeyboardInterrupt:
        pass
        # save_model(iteration)
        # summer.close()
