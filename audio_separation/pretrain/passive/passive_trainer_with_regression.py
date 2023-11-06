import os
import logging
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from habitat import logger

from audio_separation.common.base_trainer import BaseRLTrainer
from audio_separation.common.baseline_registry import baseline_registry
from audio_separation.common.tensorboard_utils import TensorboardWriter
from audio_separation.pretrain.passive.policy import PassiveSepPolicy
from audio_separation.pretrain.passive.passive import Passive
from audio_separation.pretrain.datasets.dataset_with_regression import PassiveDataset
from habitat_audio.utils import load_points_data




import torch
import torch.nn as nn
import os
import copy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import torchaudio
import torchaudio.functional as F
# import torchaudio.transforms as T

from scipy.io import wavfile
from scipy.signal import fftconvolve
from torchvision import datasets, models
from torchvision import transforms as T

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch.optim as optim

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy import signal
from torch.utils.tensorboard import SummaryWriter


SCENE_SPLITS = {
    "mp3d":
        {
            'train':
                ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
                 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
                 'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va',  '5LpN3gDmAk7',
                 'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
                 'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
                 'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
                 '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                 'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
                 'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
                 '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
            'val':
                ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
                 'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
        },
}
'''
SCENE_SPLITS = {
    "mp3d":
        {
            'train':
                ['B6ByNegPMKs', 'kEZ7cmS4wCh'],
            'val':
                ['x8F5xyUWy9e', 'QUCTc6BB5sX'],
        },
}
'''

EPS = 1e-7


flag = False


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer for audio spects..

    Takes in separated audio outputs (bin/monos) and produces an embedding

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        encode_monoFromMem: creates CNN for encoding predicted monaural from transformer memory if set to True
    """
    def __init__(self, observation_space=None, output_size=2, encode_monoFromMem=False, encoder_type='cnn'):
        super().__init__()
        self.encode_monoFromMem = encode_monoFromMem
        # originally 2 channels for binaural and 1 channel for mono but spec. sliced up into 16 chunks along the frequency
        # dimension (this makes the high-res. specs. easier to deal with)
        self._slice_factor = 16

        self._n_input_audio = 1 if encode_monoFromMem else 2
        self._n_input_audio *= self._slice_factor

        self.encoder_type = encoder_type

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (2, 2)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        cnn_dims = np.array(
            [512 // 16,
             32],
            dtype=np.float32
        )

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        if 'resnet' in self.encoder_type:
            print("Initializing Resnet")
            self.rn =  models.resnet18(pretrained=False)
            self.rn.conv1 = nn.Conv2d(2, 64, kernel_size=8, stride=4, padding=3, bias=False)
            num_ftrs = self.rn.fc.in_features
            self.rn.fc = nn.Linear(num_ftrs, 2)

        else:
            print("Initializing Simple 3 layer CNN")
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_audio,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                nn.ReLU(True),
                Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.Tanh(),
            )

            self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations, pred_binSepMasks=None, pred_monoFromMem=None,):
        cnn_input = []

        x = observations

        # slice along freq dimension into 16 chunks
        # x = x.view(x.size(0), x.size(1), self._slice_factor, -1, x.size(3))
        # x = x.reshape(x.size(0), -1, x.size(3),  x.size(4))

        cnn_input.append(x)
        cnn_input = torch.cat(cnn_input, dim=1)

        if 'resnet' in self.encoder_type:
            return self.rn(cnn_input)
        else:
            return self.cnn(cnn_input)


@baseline_registry.register_trainer(name="passive_loc")
class PassiveTrainerWithRegression(BaseRLTrainer):
    r"""Trainer class for pretraining passive separators in a supervised fashion
    """

    def __init__(self, config=None):
        super().__init__(config, None)
        self.actor_critic = None
        self.agent = None

    def _setup_passive_agent(self,) -> None:
        r"""Sets up agent for passive pretraining.
        Args:
            None
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = PassiveSepPolicy()

        self.actor_critic.to(self.device)
        self.actor_critic.train()

        self.agent = Passive(
            actor_critic=self.actor_critic,
        )

    def save_checkpoint(self, file_name: str,) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def get_dataloaders(self):
        r"""
        build datasets and dataloaders
        :return:
            dataloaders: PyTorch dataloaders for training and validation
            dataset_sizes: sizes of train and val datasets
        """
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        audio_cfg = sim_cfg.AUDIO

        scene_splits = {"train": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["train"],
                        # "val": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["val"],
                        "nonoverlapping_val": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["val"]}
        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        print("num_worker is ", audio_cfg.NUM_WORKER)
        
        # import pdb; pdb.set_trace()
        
        for split in scene_splits:
            scenes = scene_splits[split]
            scene_graphs = dict()
            for scene in scenes:
                try:
                  _, graph = load_points_data(
                      os.path.join(audio_cfg.META_DIR, scene),
                      audio_cfg.GRAPH_FILE,
                      transform=True,
                      scene_dataset=sim_cfg.SCENE_DATASET)
                  scene_graphs[scene] = graph
                except:
                  pass
                # import pdb; pdb.set_trace()

            print("scene_graph.keys ", scene_graphs.keys())
            datasets[split] = PassiveDataset(
                split=split,
                scene_graphs=scene_graphs,
                sim_cfg=sim_cfg,
            )

            dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=audio_cfg.BATCH_SIZE,
                                            shuffle=(split == 'train'),
                                            pin_memory=True,
                                            num_workers=audio_cfg.NUM_WORKER,
                                            )
            #import pdb; pdb.set_trace()
            
            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))
            
        # to be deleted
        # dataloaders['val'] = dataloaders['train']
        # dataset_sizes['val'] = dataset_sizes['train']
            
        return dataloaders, dataset_sizes

    def train(self) -> None:
        r"""Main method for training passive separators using supervised learning.
        Returns:
            None
        """
        passive_cfg = self.config.Pretrain.Passive

        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_passive_agent()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        root_dir = 'data/active_datasets/v1_old/train_731243episodes/regression_resnet_5nov_lr_1e-4_l1_factor1_pretrainedSep'
        encoder_type='resnet' # choices are 'cnn' or 'resnet'. 'cnn' will invoke simple CNN
        device_ids = [0,1,2,3] # for 4 gpus

        os.makedirs(root_dir, exist_ok = True)
        model = AudioCNN(output_size=2,  encoder_type=encoder_type)

        model = nn.DataParallel(model, device_ids = device_ids) 

        criterion = torch.nn.L1Loss()    # torch.nn.MSELoss()    
        optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-8)

        try:
            val_checkpoint = torch.load(root_dir + '/last_ckpt.pth')
            model.load_state_dict(val_checkpoint['state_dict'])
            optimizer.load_state_dict(val_checkpoint['optimizer'])
            start_epoch = val_checkpoint['epoch'] + 1
            print("resuming from checkpoint: ", root_dir)
        except:
            start_epoch = 0
            print("Starting new training with foldername: ", root_dir)
            print("")
        model = model.to(device)

        tb_log_subdir = "tb" 
        tb_log_dir = os.path.join(root_dir, "tb")
        if os.path.isdir(tb_log_dir):
            for i in range(1, 10000):
                tb_log_dir_2 = os.path.join(root_dir, f"tb_{i}")
                if not os.path.isdir(tb_log_dir_2):
                    os.system(f"mv {tb_log_dir} {tb_log_dir_2}")
                    break

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
        #                              lr=passive_cfg.lr, eps=passive_cfg.eps)

        print('Optimizer initialised')
        # build datasets and dataloaders
        dataloaders, dataset_sizes = self.get_dataloaders()
        print('Dloader init')

        passive_ckpt_path = "/fs/nexus-projects/ego_data/active_avsep/active-AV-dynamic-separation/runs/passive_pretrain/new/data/best_ckpt_nonoverlapping_val.pth"   # PUT THE PASSIVE CKPT PATH HERE
        try:
            self.agent.load_state_dict(torch.load(passive_ckpt_path, map_location="cpu")["state_dict"])
            self.actor_critic = self.agent.actor_critic
            print('Actor critic ckpt loaded')
        except:
            raise NotImplementedError()

        best_mono_loss = float('inf')
        best_nonoverlapping_mono_loss = float('inf')
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        print('start tboard')
        with TensorboardWriter(
            tb_log_dir, flush_secs=self.flush_secs
        ) as writer:
            print('Inside tb writer')
            for epoch in range(start_epoch, self.config.NUM_EPOCHS):    #for epoch in range(self.config.NUM_EPOCHS):
                logging.info('-' * 10)
                # import pdb; pdb.set_trace()
                logging.info('Epoch {}/{}'.format(epoch, self.config.NUM_EPOCHS - 1))
                for split in dataloaders.keys():
                    if split == "train":
                        self.actor_critic.eval()
                        model.train()
                    else:
                        self.actor_critic.eval()
                        model.eval()

                    running_loss = 0.0
                    val_loss = 0.0
                    val_l1_loss = 0.0
                    for i, data in enumerate(tqdm(dataloaders[split],desc="inside training loop")):
                        #import pdb; pdb.set_trace()
                        mixed_audio = data[0].to(self.device)
                        gt_bin_mag = data[1].to(self.device)[..., 0:2]
                        gt_mono_mag = data[2].to(self.device)[..., :1]
                        target_class = data[3].to(self.device)
                        delta_x = data[4].to(self.device)
                        delta_y = data[5].to(self.device)
                        labels = torch.stack((delta_x,delta_y), axis=-1)
                        bs = target_class.size(0)
                        optimizer.zero_grad()

                        obs_batch = {"mixed_bin_audio_mag": mixed_audio, "target_class": target_class}

                        with torch.no_grad():
                            pred_binSepMasks = self.actor_critic.get_binSepMasks(obs_batch)
                            pred_binSep = pred_binSepMasks * (torch.exp(mixed_audio) - 1).detach()
                            # pred_mono =\
                            #     self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                            #                                            mixed_audio=mixed_audio)

                        with torch.set_grad_enabled(split == 'train'):
                            #   inputs = (inputs - torch.min(inputs)) / (torch.max(inputs) - torch.min(inputs))
                            outputs = model(pred_binSep.permute(0,3,1,2)) # input shape should be torch.rand(2,1,512,32)
                            #   import pdb; pdb.set_trace()
                            loss = criterion(outputs*1., labels*1.)

                            if split == 'train':
                                # print("@!@!@!INSIDE TRAIN")
                                loss.backward()
                                optimizer.step()
                            else:
                                if flag:
                                    l1_loss = torch.nn.L1Loss()(outputs*20., labels*20.)
                                else:
                                    l1_loss = torch.nn.L1Loss()(outputs*1., labels*1.)

                        if split == 'train':
                            running_loss += loss.item() * bs
                        else:
                            val_loss += loss.item() * bs # added for validation
                            val_l1_loss += l1_loss.item() * bs

                    phase = split
                    dsets = dataset_sizes
                    if phase == 'train':
                        epoch_train_loss = running_loss / dataset_sizes[phase]
                        print(f'{phase} Loss: {epoch_train_loss:.4f}')

                        writer.add_scalar('train_loss', epoch_train_loss, epoch)


                        if epoch_train_loss < best_train_loss:
                            best_train_loss = epoch_train_loss
                            # torch.save({'state_dict':model.state_dict()},root_dir + "/best_train_ckpt.pth")
                            torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_loss": best_train_loss},root_dir + "/best_train_ckpt.pth")

                    else:
                        epoch_val_loss = val_loss /  dataset_sizes[phase] # added for validation
                        epoch_val_l1_loss = val_l1_loss / dataset_sizes[phase] # added for validation

                        writer.add_scalar('val_loss', epoch_val_loss, epoch)
                        writer.add_scalar('val_L1_loss', epoch_val_l1_loss, epoch)

                        print(f'{phase} Loss: {epoch_val_loss:.4f}')
                        print(f'{phase} L1 Loss: {epoch_val_l1_loss:.4f}')

                        if epoch_val_loss < best_val_loss:
                            best_val_loss = epoch_val_loss
                            # torch.save({'state_dict':model.state_dict()},root_dir + "/best_val_ckpt.pth")
                            torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_loss": best_val_loss},root_dir + "/best_val_ckpt.pth")

                    torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss},root_dir + "/last_ckpt.pth")

    def optimize_supervised_loss(self, optimizer, mixed_audio, pred_binSepMasks, gt_bin_mag, pred_mono, gt_mono_mag,
                                 split='train',):
        mixed_audio = torch.exp(mixed_audio) - 1
        pred_bin = pred_binSepMasks * mixed_audio
        bin_loss = F.l1_loss(pred_bin, gt_bin_mag)

        mono_loss = F.l1_loss(pred_mono, gt_mono_mag)

        if split == "train":
            optimizer.zero_grad()
            loss = bin_loss + mono_loss
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.Pretrain.Passive.max_grad_norm
            )
            loss.backward()
            optimizer.step()

        return bin_loss, mono_loss, optimizer
