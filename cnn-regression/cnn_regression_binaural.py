
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



# class BinauralRIRdataset(Dataset):

#     def __init__(self, filename='train_wavs.json', split='train'):

#         self.filename = filename
#         self.split = split
#         f = open(filename, 'rb')
#         self.list_wavs = json.load(f)
#         self.list_wavs = self.list_wavs[self.split]
#         self.image_transform = T.Compose([T.Resize(256),T.RandomCrop(224),T.ToTensor()]) # check transforms

#     def __len__(self):
#         return len(self.list_wavs)

#     def __getitem__(self, idx):

#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         bin_rir_fname = self.list_wavs[idx]['binaural_rir_filename'] # please check the keys here from the list of dicts
#         target_xy = np.asarray(self.list_wavs[idx]['target']) # please check the keys here from the list of dicts

#         waveform, sr = torchaudio.load(bin_rir_fname)

#         # if waveform.shape[1] != 0:
#         #     waveform = waveform / torch.max(waveform)
#         try:
#             specgram = torchaudio.transforms.Spectrogram()(waveform).log2()
#             # except:
#                 # print("waveform.shape", waveform.shape)
#                 # print(abcd)
#             # specgram = specgram.unsqueeze(0) # shape is of the form torch.Size([1, 1, 201, 331]). You may have to repeat to make 3 channel.
#             # Verify is unsqueeze is needed or not

#             # print("specgram shape is ", specgram.shape)
#             specgram = specgram[0,:,:]
#             specgram = specgram.numpy()
#             # print("specgram shape before transform is ", specgram.shape)
#             specgram =  Image.fromarray(specgram)
#             spectrogram = self.image_transform(specgram)
#             spectrogram = spectrogram.repeat(3,1,1)

#             if torch.isnan(torch.mean(spectrogram)) or torch.isinf(torch.mean(spectrogram)):
#                 spectrogram = torch.ones(3,224,224)
#             # print(torch.max(spectrogram), torch.min(spectrogram))
#         except:
#             spectrogram = torch.ones(3,224,224)

#         # print("spectrogram shape after transform is ", spectrogram.shape)
#         # print("torch.from_numpy(target_xy).shape is ", torch.from_numpy(target_xy).shape)
#         return spectrogram, torch.from_numpy(target_xy) / 50.


# dsets = dict()
# dsets['train'] = BinauralRIRdataset(filename='train_wavs.json', split='train')
# dsets['val'] = BinauralRIRdataset(filename='val_wavs.json', split='val') # added for validation

# dataloaders = dict()
# dataloaders['train'] = DataLoader(dsets['train'], batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
# # print("done!!!")
# dataloaders['val'] = DataLoader(dsets['val'], batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) # added for validation

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model =  models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)
# model = model.to(device)

# criterion = torch.nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# num_epochs = 100
# best_val_loss = 10000000.
# best_train_loss = 10000000.

# for _,epoch in enumerate(tqdm(range(num_epochs))):
#   print(f'Epoch {epoch}/{num_epochs - 1}')
#   print('-' * 10)

#   for phase in ['train','val']:
#       if phase == 'train':
#           model.train()
#       else: # added for validation
#           model.eval() # added for validation

#       running_loss = 0.0
#       val_loss = 0.0

#       for inputs, labels in dataloaders[phase]:
#           inputs = inputs.float().to(device)
#           labels = labels.float().to(device)

#           optimizer.zero_grad()

#           with torch.set_grad_enabled(phase == 'train'):
#               inputs = (inputs - torch.min(inputs)) / (torch.max(inputs) - torch.min(inputs))
#               outputs = model(inputs)
#             #   import pdb; pdb.set_trace()
#               loss = criterion(outputs, labels)

#               if phase == 'train':
#                   loss.backward()
#                   optimizer.step()


#           if phase == 'train':
#               running_loss += loss.item() * inputs.size(0)
#           else:
#               val_loss += loss.item() * inputs.size(0) # added for validation

#       if phase == 'train':
#           epoch_train_loss = running_loss / len(dataloaders[phase])
#           print(f'{phase} Loss: {epoch_train_loss:.4f}')
#           if epoch_train_loss < best_train_loss:
#               best_train_loss = epoch_train_loss
#               torch.save({'state_dict':model.state_dict()},"rir_regression_ckpt/best_train_ckpt.pth")
#       else:
#           epoch_val_loss = val_loss / len(dataloaders[phase]) # added for validation
#           print(f'{phase} Val Loss: {epoch_val_loss:.4f}')
#           if epoch_val_loss < best_val_loss:
#               best_val_loss = epoch_val_loss
#               torch.save({'state_dict':model.state_dict()},"rir_regression_ckpt/best_val_ckpt.pth")


# **********************************************************************************************

flag = True  #False


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

class BinauralRIRdataset(Dataset):

    def __init__(self,
                 filename='train_wavs.json',
                 split='train',
                 use_mic_noise=False,
                 mic_noise_level=15,
                 ):

        self.filename = filename
        self.split = split
        f = open(filename, 'rb')
        self.list_wavs = json.load(f)
        self.list_wavs = self.list_wavs[self.split]
        # self.image_transform = T.Compose([T.Resize((512,32)),T.ToTensor()]) # check transforms  T.RandomCrop(224), # please check transformation size here output should be (512, 32)

        self.split = split
        self.use_mic_noise = use_mic_noise
        self.mic_noise_level = mic_noise_level

        if split in ["val"]:
            if use_mic_noise:
                self.cached_noise_sample = np.random.normal(0, 1, (len(self.list_wavs), 2, 16000))  # self.cached_noise_sample = np.random.normal(0, 1, (len(self.list_wavs), 2, 1))

    def __len__(self):
        return  len(self.list_wavs[:])    #:, :6

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        bin_rir_fname = self.list_wavs[idx]['binaural_rir_filename'] # please check the keys here from the list of dicts
        target_xy = np.asarray(self.list_wavs[idx]['target']) # please check the keys here from the list of dicts
        mono_fname = self.list_wavs[idx]['mono_filename']

        assert os.path.isfile(bin_rir_fname)
        assert os.path.isfile(mono_fname)

        folder_name = 'precomputed_data'
        gt_bin_mono_fname = folder_name + '/' + bin_rir_fname.replace("/", "_")[:-4].split("binaural_rirs")[-1] + '_' + mono_fname.replace("/", "_")[:-4].split("audio_data")[-1] + '.pt'
        # print("@!@!@!@!@!gt_bin_mono_fname ", gt_bin_mono_fname)
        # assert False


        # if not os.path.exists(gt_bin_mono_fname):

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

        if self.use_mic_noise:
            rms_signal = np.power(np.mean(np.power(binaural_convolved, 2, dtype="float32")), 0.5)
            noise_sigma = rms_signal / np.power(10, (self.mic_noise_level / 20))
            # noise = np.random.normal(0, noise_sigma, binaural_convolved.shape)
            if self.split in ["val"]:
                # print("h1")
                noise = self.cached_noise_sample[idx] * noise_sigma
            else:
                # print("h2")
                # noise = torch.normal(0, 1, size = (2,1)).numpy() * noise_sigma    # make size = binaural_convolved.shape  OR  (2,1)
                noise = np.random.normal(0, 1, (2,1)) * noise_sigma
                # print("@!@!@!h2: ", noise)
            binaural_convolved += noise

        # compute gt bin. magnitude
        fft_windows_l = librosa.stft(np.asfortranarray(binaural_convolved[0]), hop_length=HOP_LENGTH,
                                        n_fft=N_FFT)
        magnitude_l, _ = librosa.magphase(fft_windows_l)

        fft_windows_r = librosa.stft(np.asfortranarray(binaural_convolved[1]), hop_length=HOP_LENGTH,
                                        n_fft=N_FFT)
        magnitude_r, _ = librosa.magphase(fft_windows_r)

        gt_bin_mag = np.stack([magnitude_l, magnitude_r], axis=-1).astype("float32")

        gt_bin_mag = torch.from_numpy(gt_bin_mag)

        #     torch.save(gt_bin_mag, gt_bin_mono_fname)

        # else:

        #     gt_bin_mag = torch.load(gt_bin_mono_fname)

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
        if flag:
            return gt_bin_mag.permute(2,0,1), torch.from_numpy(target_xy) / 20. # 50.
        else:
            return gt_bin_mag.permute(2,0,1), torch.from_numpy(target_xy) / 1. # 50. 


USE_MIC_NOISE = True    # True, False
MIC_NOISE_LEVEL_IN_DB = 60  # 15, 30, 45, 60

DATASET_VERSION = 1
BATCH_SIZE = 512  # 512
NUM_WORKERS = 4 # 4

NUM_EPOCHS = 100
LR = 1e-4   # 1e-4, 1e-5

ENCODER_TYPE = "resnet"

""" run dirs copied from active training config: regression_resnet_filtered_30oct_rectified_40_lr_1e-4_l1_factor1 (probably used), regression_resnet_filtered_30oct_rectified_40_lr_1e-4_l1_factor1  """
RUN_DRNM = "lr1e4_60dBnoise"  # "regression_resnet_filtered_2nov_rectified_20_lr_1e-4_l1_factor1_micNoise15_updated"


DATASET_ROOT_DR = f"data/passive_datasets/v{DATASET_VERSION}"
RUNS_ROOT_DR = f"runs/passivePretrain_locationPredictor"


RUN_DR = f"{RUNS_ROOT_DR}/{RUN_DRNM}"
if not os.path.isdir(RUN_DR):
    os.makedirs(RUN_DR)


dsets = dict()
dsets['train'] = BinauralRIRdataset(filename=f'{DATASET_ROOT_DR}/train_locationPredictor/allepisodesPerScene.json',
                                    split='train', 
                                    use_mic_noise=USE_MIC_NOISE,
                                    mic_noise_level=MIC_NOISE_LEVEL_IN_DB)
dsets['val'] = BinauralRIRdataset(filename=f'{DATASET_ROOT_DR}/val_locationPredictor/allepisodesPerScene.json',
                                  split='val',
                                  use_mic_noise=USE_MIC_NOISE,
                                  mic_noise_level=MIC_NOISE_LEVEL_IN_DB,) # added for validation

dataloaders = dict()
dataloaders['train'] = DataLoader(dsets['train'], 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=NUM_WORKERS, 
                                  pin_memory=True, 
                                  drop_last=False)
# print("done!!!")
dataloaders['val'] = DataLoader(dsets['val'], 
                                batch_size=BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=NUM_WORKERS, 
                                pin_memory=True, 
                                drop_last=False) # added for validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_available_gpus = torch.cuda.device_count()
device_ids = list(range(n_available_gpus))  # [0,1,2,3]: for 4 gpus

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)

root_dir = f"{RUNS_ROOT_DR}/{RUN_DRNM}"
encoder_type= ENCODER_TYPE # choices are 'cnn' or 'resnet'. 'cnn' will invoke simple CNN

os.makedirs(root_dir, exist_ok = True)
model = AudioCNN(output_size=2,  encoder_type=encoder_type)

model = nn.DataParallel(model, device_ids = device_ids)

criterion = torch.nn.L1Loss()    # torch.nn.MSELoss()    
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-8)

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
# print("loaded checkpoint successfully!!")

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

num_epochs = NUM_EPOCHS
# best_val_loss = 10000000.
# best_train_loss = 10000000.

try:
    best_val_loss = val_checkpoint['best_val_loss']
    best_train_loss = val_checkpoint['best_train_loss']
except:
    best_val_loss = float('inf')
    best_train_loss = float('inf')


tb_log_subdir = "tb" 
tb_log_dir = os.path.join(root_dir, "tb")

if os.path.isdir(tb_log_dir):
    for i in range(1, 10000):
        tb_log_dir_2 = os.path.join(root_dir, f"tb_{i}")
        if not os.path.isdir(tb_log_dir_2):
            os.system(f"mv {tb_log_dir} {tb_log_dir_2}")
            break


assert not os.path.isdir(tb_log_dir) 
os.makedirs(tb_log_dir)

writer = SummaryWriter(log_dir=tb_log_dir)


print("len of train dataloader ", len(dataloaders['train']))
print("len of val dataloader ", len(dataloaders['val']))

for _,epoch in enumerate(tqdm(range(start_epoch,num_epochs))):
  print(f'Epoch {epoch}/{num_epochs - 1}')
  print('-' * 10)

  for phase in ['train', 'val']:
      target_max = -10000.
      target_min = 10000.

      if phase == 'train':
          model.train()
      else: # added for validation
          model.eval() # added for validation

      running_loss = 0.0
      val_loss = 0.0
      val_l1_loss = 0.0

      for inputs_n_labels in tqdm(dataloaders[phase]):   # for _, (inputs, labels) in enumerate(dataloaders[phase]):
          inputs = inputs_n_labels[0]
          labels = inputs_n_labels[1]

          inputs = inputs.float().to(device)
          labels = labels.float().to(device)

          if target_max < labels.max().item():
              target_max = labels.max().item()

          if target_min > labels.min().item():
              target_min = labels.min().item()

          optimizer.zero_grad()

          with torch.set_grad_enabled(phase == 'train'):
            #   inputs = (inputs - torch.min(inputs)) / (torch.max(inputs) - torch.min(inputs))
              outputs = model(inputs) # input shape should be torch.rand(2,1,512,32)
            #   import pdb; pdb.set_trace()
              loss = criterion(outputs*1., labels*1.)

              if phase == 'train':
                  loss.backward()
                  optimizer.step()
              else:
                if flag:
                    l1_loss = torch.nn.L1Loss()(outputs*20., labels*20.)
                else:
                    l1_loss = torch.nn.L1Loss()(outputs*1., labels*1.)

                        

          if phase == 'train':
              running_loss += loss.item() * inputs.size(0)
          else:
              val_loss += loss.item() * inputs.size(0) # added for validation
              val_l1_loss += l1_loss.item() * inputs.size(0)

      if phase == 'train':
          epoch_train_loss = running_loss / len(dsets[phase])
          print(f'{phase} Loss: {epoch_train_loss:.4f}')
          print(f'{phase} target max: {target_max:.4f}')
          print(f'{phase} target min: {target_min:.4f}')

          writer.add_scalar('train_loss', epoch_train_loss, epoch)


          if epoch_train_loss < best_train_loss:
              best_train_loss = epoch_train_loss
              # torch.save({'state_dict':model.state_dict()},root_dir + "/best_train_ckpt.pth")
              torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_loss": best_train_loss},root_dir + "/best_train_ckpt.pth")

      else:
          epoch_val_loss = val_loss / len(dsets[phase]) # added for validation
          epoch_val_l1_loss = val_l1_loss / len(dsets[phase]) # added for validation

          writer.add_scalar('val_loss', epoch_val_loss, epoch)
          writer.add_scalar('val_L1_loss', epoch_val_l1_loss, epoch)

          print(f'{phase} Loss: {epoch_val_loss:.4f}')
          print(f'{phase} L1 Loss: {epoch_val_l1_loss:.4f}')
          print(f'{phase} target max: {target_max:.4f}')
          print(f'{phase} target min: {target_min:.4f}')
          if epoch_val_loss < best_val_loss:
              best_val_loss = epoch_val_loss
              # torch.save({'state_dict':model.state_dict()},root_dir + "/best_val_ckpt.pth")
              torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_loss": best_val_loss},root_dir + "/best_val_ckpt.pth")

      torch.save({'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss},root_dir + "/last_ckpt.pth")    

writer.close()


# CUDA_VISIBLE_DEVICES=0,1,2,3 python cnn_regression_binaural.py
