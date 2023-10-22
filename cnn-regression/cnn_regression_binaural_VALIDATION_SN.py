
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
            self.rn =  models.resnet18(pretrained=True)
            self.rn.conv1 = nn.Conv2d(self._n_input_audio, 64, kernel_size=8, stride=4, padding=3, bias=False)
            num_ftrs = self.rn.fc.in_features
            self.rn.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Tanh())

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
        x = x.view(x.size(0), x.size(1), self._slice_factor, -1, x.size(3))
        x = x.reshape(x.size(0), -1, x.size(3),  x.size(4))

        cnn_input.append(x)
        cnn_input = torch.cat(cnn_input, dim=1)

        if 'resnet' in self.encoder_type:
            return self.rn(cnn_input)
        else:
            return self.cnn(cnn_input)

class BinauralRIRdataset(Dataset):

    def __init__(self, filename='train_wavs.json', split='train'):

        self.filename = filename
        self.split = split
        f = open(filename, 'rb')
        self.list_wavs = json.load(f)
        self.list_wavs = self.list_wavs[self.split]
        # self.image_transform = T.Compose([T.Resize((512,32)),T.ToTensor()]) # check transforms  T.RandomCrop(224), # please check transformation size here output should be (512, 32)

    def __len__(self):
        return len(self.list_wavs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        bin_rir_fname = self.list_wavs[idx]['binaural_rir_filename'] # please check the keys here from the list of dicts
        target_xy = np.asarray(self.list_wavs[idx]['target']) # please check the keys here from the list of dicts
        mono_fname = self.list_wavs[idx]['mono_filename']

        folder_name = 'precomputed_data'
        gt_bin_mono_fname = folder_name + '/' + bin_rir_fname.replace("/", "_")[:-4] + '_' + mono_fname.replace("/", "_")[:-4] + '.pt'

        if not os.path.exists(gt_bin_mono_fname):

            HOP_LENGTH = 512
            N_FFT = 1023
            rir_sampling_rate = 16000
            try:
                sr, binaural_rir = wavfile.read(bin_rir_fname)
            except ValueError:
                binaural_rir = np.zeros((rir_sampling_rate, 2)).astype("float32")
                sr = rir_sampling_rate
            if len(binaural_rir) == 0:
                binaural_rir = np.zeros((rir_sampling_rate, 2)).astype("float32")

            try:
              sr_mono, mono_audio = wavfile.read(mono_fname)
              # signal.resample(waveform.numpy()[:, 0:1],sr).T
            except:
              sr_mono, mono_audio = sr, binaural_rir[:,0]

            binaural_convolved = []

            for channel in range(binaural_rir.shape[-1]):
                binaural_convolved.append(fftconvolve(mono_audio, binaural_rir[:, channel], mode="same"))

            binaural_convolved = np.array(binaural_convolved)
            # this makes sure that the audio is in the range [-32768, 32767]
            binaural_convolved = np.round(binaural_convolved).astype("int16").astype("float32")
            binaural_convolved *= (1 / 32768)

            if binaural_convolved.shape[1] >= 16000:
                binaural_convolved = binaural_convolved[:,:16000]
            else:
                binaural_convolved =  np.concatenate((binaural_convolved,np.zeros((binaural_convolved.shape[0], 16000 - binaural_convolved.shape[1]))), axis = 1)

            # compute gt bin. magnitude
            fft_windows_l = librosa.stft(np.asfortranarray(binaural_convolved[0]), hop_length=HOP_LENGTH,
                                          n_fft=N_FFT)
            magnitude_l, _ = librosa.magphase(fft_windows_l)

            fft_windows_r = librosa.stft(np.asfortranarray(binaural_convolved[1]), hop_length=HOP_LENGTH,
                                          n_fft=N_FFT)
            magnitude_r, _ = librosa.magphase(fft_windows_r)

            gt_bin_mag = np.stack([magnitude_l, magnitude_r], axis=-1).astype("float32")

            gt_bin_mag = torch.from_numpy(gt_bin_mag)

            torch.save(gt_bin_mag, gt_bin_mono_fname)

        else:

            gt_bin_mag = torch.load(gt_bin_mono_fname)

        assert gt_bin_mag.shape[1] == 32

        '''
        if gt_bin_mag.shape[1] > 32:
            gt_bin_mag = gt_bin_mag[:,:32,:]
        elif gt_bin_mag.shape[1] == 32:
            pass
        else:
            dim_1 = gt_bin_mag.shape[1]
            dummy_var = np.zeros((gt_bin_mag.shape[0], 32 - dim_1, gt_bin_mag.shape[2])).astype("float32")
            gt_bin_mag = np.concatenate((gt_bin_mag, dummy_var), axis=1)
        '''

        return gt_bin_mag.permute(2,0,1), torch.from_numpy(target_xy) / 20. # 50.



dsets = dict()
# dsets['train'] = BinauralRIRdataset(filename='train_wavs.json', split='train')
dsets['val'] = BinauralRIRdataset(filename='val_wavs.json', split='val') # added for validation

dataloaders = dict()
# dataloaders['train'] = DataLoader(dsets['train'], batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
# print("done!!!")
dataloaders['val'] = DataLoader(dsets['val'], batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) # added for validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)

MAX_VAL = 1.0 # set the value = target max of training labels

root_dir = "rir_regression_ckpt_lr_0.0001_tanh_resnet"
encoder_type='resnet' # choices are 'cnn' or 'resnet'. 'cnn' will invoke simple CNN
device_ids = [0,1,2,3,4,5,6,7] # for 8 gpus

os.makedirs(root_dir, exist_ok = True)
model = AudioCNN(output_size=2,  encoder_type=encoder_type)

model = nn.DataParallel(model, device_ids = device_ids)

try:
    model.load_state_dict(torch.load(root_dir + '/best_val_ckpt.pth')['state_dict'])
    print("resuming from checkpoint!!")
except:
    print("Starting new training!!")
model = model.to(device)
# print("loaded checkpoint successfully!!")

criterion = torch.nn.L1Loss() # torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-8)

num_epochs = 100
best_val_loss = 10000000.
best_train_loss = 10000000.

writer = SummaryWriter()

for _,epoch in enumerate(tqdm(range(num_epochs))):
  print(f'Epoch {epoch}/{num_epochs - 1}')
  print('-' * 10)

  for phase in ['val']:
      target_max = -10000.
      target_min = 10000.

      if phase == 'train':
          model.train()
      else: # added for validation
          model.eval() # added for validation

      running_loss = 0.0
      val_loss = 0.0
      val_l1_loss = 0.0

      for _, (inputs, labels) in enumerate(dataloaders[phase]):
          inputs = inputs.float().to(device)
          labels = labels.float().to(device)

          if target_max < labels.max().item():
              target_max = labels.max().item()

          if target_min > labels.min().item():
              target_min = labels.min().item()

          optimizer.zero_grad()

          # labels shape = (BATCH_SIZE, 2) # verify this
          multiplier = torch.ones(labels.shape[0],1).to(labels.device)
          for _iter in range(labels.shape[0]):
              abs_del_x, abs_del_y = torch.abs(labels[_iter,0]), torch.abs(labels[_iter,1])
              if torch.max(abs_del_x, abs_del_y) > MAX_VAL:
                  multiplier[_iter] *= 0.

          with torch.set_grad_enabled(phase == 'train'):
            #   inputs = (inputs - torch.min(inputs)) / (torch.max(inputs) - torch.min(inputs))
              outputs = model(inputs) # input shape should be torch.rand(2,1,512,32)
            #   import pdb; pdb.set_trace()
              outputs = outputs * multiplier
              labels = labels * multiplier
              loss = criterion(outputs*20., labels*20.)

              if phase == 'train':
                  loss.backward()
                  optimizer.step()
              else:
                  l1_loss = torch.nn.L1Loss()(outputs*20., labels*20.)

          if phase == 'train':
              running_loss += loss.item() * inputs.size(0)
          else:
              val_loss += loss.item() * inputs.size(0) # added for validation
              val_l1_loss += l1_loss.item() * inputs.size(0)

      if phase == 'train':
          epoch_train_loss = running_loss / len(dataloaders[phase])
          print(f'{phase} Loss: {epoch_train_loss:.4f}')
          print(f'{phase} target max: {target_max:.4f}')
          print(f'{phase} target min: {target_min:.4f}')

          writer.add_scalar('train_loss', epoch_train_loss, epoch)


          if epoch_train_loss < best_train_loss:
              best_train_loss = epoch_train_loss
              torch.save({'state_dict':model.state_dict()},root_dir + "/best_train_ckpt.pth")
      else:
          epoch_val_loss = val_loss / len(dataloaders[phase]) # added for validation
          epoch_val_l1_loss = val_l1_loss / len(dataloaders[phase]) # added for validation

          writer.add_scalar('val_loss', epoch_val_loss, epoch)
          writer.add_scalar('val_L1_loss', epoch_val_l1_loss, epoch)

          print(f'{phase} Loss: {epoch_val_loss:.4f}')
          print(f'{phase} L1 Loss: {epoch_val_l1_loss:.4f}')
          print(f'{phase} target max: {target_max:.4f}')
          print(f'{phase} target min: {target_min:.4f}')
          # if epoch_val_loss < best_val_loss:
          #     best_val_loss = epoch_val_loss
          #     torch.save({'state_dict':model.state_dict()},root_dir + "/best_val_ckpt.pth")

writer.close()
