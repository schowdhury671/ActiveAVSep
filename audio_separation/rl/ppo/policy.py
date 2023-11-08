import abc

import torch
import torch.nn as nn

from audio_separation.common.utils import CategoricalNet
from audio_separation.rl.models.rnn_state_encoder import RNNStateEncoder
from audio_separation.rl.models.visual_cnn import VisualCNN
from audio_separation.rl.models.audio_cnn import AudioCNN
from audio_separation.rl.models.separator_cnn import PassiveSepEncCNN, PassiveSepDecCNN
from audio_separation.rl.models.memory_nets import AudioMem
from audio_separation.rl.models.location_predictor import AudioCNN as LocationCNN


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PolicyNet(Net):
    r"""Network which passes the observations and separated audio outputs through CNNs and concatenates
    them into a single vector before passing that through RNN.
    """
    def __init__(self, observation_space, hidden_size, extra_rgb=False, extra_depth=False, use_location_in_policy=False,
                 use_predictedLocation_in_policy=False, use_predictedLocationWoFeats_in_policy=False, locationPredictor_encoderType="resnet"):
        super().__init__()

        self._use_location_in_policy = use_location_in_policy
        self.use_predictedLocation_in_policy = use_predictedLocation_in_policy
        self.use_predictedLocationWoFeats_in_policy = use_predictedLocationWoFeats_in_policy
        self._hidden_size = hidden_size

        if use_predictedLocation_in_policy:
            assert use_location_in_policy

        if use_predictedLocationWoFeats_in_policy:
            assert use_predictedLocation_in_policy

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb, extra_depth)
        self.bin_encoder = AudioCNN(observation_space, hidden_size)
        self.monoFromMem_encoder = AudioCNN(observation_space, hidden_size, encode_monoFromMem=True,)

        if self._use_location_in_policy:
            if use_predictedLocation_in_policy:
                if use_predictedLocationWoFeats_in_policy:
                    self.location_encoder = LocationCNN(output_size=2,  encoder_type=locationPredictor_encoderType, return_feats=False,)
                    self.location_encoder_linear = nn.Sequential(nn.Linear(2, self._hidden_size), nn.ReLU(),
                                                      nn.Linear(self._hidden_size, self._hidden_size))
                else:
                    self.location_encoder = LocationCNN(output_size=2,  encoder_type=locationPredictor_encoderType, return_feats=True,)
            else:
                self.location_encoder = nn.Sequential(nn.Linear(2, self._hidden_size), nn.ReLU(),
                                                      nn.Linear(self._hidden_size, self._hidden_size))
            rnn_input_size = 4 * self._hidden_size
        else:
            rnn_input_size = 3 * self._hidden_size
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self,
                observations,
                rnn_hidden_states,
                masks,
                pred_binSepMasks,
                pred_monoFromMem):
        x = []
        # print("@@@@@@@@@@@@@observations keys ", observations.keys())
        # print("@@@@@@@@@@@@@observations rgb shape ", observations["rgb"].shape)

        temp_vis_enc = self.visual_encoder(observations)
        x.append(temp_vis_enc)
        # print("@@@@@@@@@@@@@vis enc shape ", temp_vis_enc.shape)
        
        temp_bin_enc = self.bin_encoder(observations, pred_binSepMasks=pred_binSepMasks)
        # print("@@@@@@@@@@@@@bin enc shape ", temp_bin_enc.shape)
        x.append(temp_bin_enc)
        
        temp_mono_enc = self.monoFromMem_encoder(observations, pred_monoFromMem=pred_monoFromMem)
        # print("@@@@@@@@@@@@@temp_mono_enc.shape ", temp_mono_enc.shape)

        x.append(temp_mono_enc)

        #x[-1] = x[-2]   # added by Sanjoy

        if self._use_location_in_policy:
            if self.use_predictedLocation_in_policy:
                assert pred_binSepMasks is not None
                inp_ = observations["mixed_bin_audio_mag"]
                inp_ = torch.exp(inp_) - 1
                # print("h0: ", inp_.shape, pred_binSepMasks.shape)
                # exit()
                inp_ = inp_ * pred_binSepMasks
                inp_ = inp_.permute((0, 3, 1, 2))
                self.location_encoder.eval()
                with torch.no_grad():
                    # print("@!@!!@!inp_.shape ", inp_.shape)
                    # print("@!@!@!@inp_ ", inp_)
                    # print(abcd)

                    temp_loc_enc = self.location_encoder(inp_)
                    temp_loc_enc = temp_loc_enc.detach()
                
                if self.use_predictedLocationWoFeats_in_policy:
                    temp_loc_enc = self.location_encoder_linear(temp_loc_enc)
                    # print("#@#@#@policy.py h1")
            else:
                temp_loc_enc = self.location_encoder(observations['ground_truth_deltax_deltay'])
            x.append(temp_loc_enc)
            # print("#@#@#@#@#@#@#@x is ", len(x))

            # print("################# temp_loc_enc.shape: ", temp_loc_enc.shape)

            del temp_loc_enc
        
        # import pdb; pdb.set_trace()
        del temp_vis_enc
        del temp_bin_enc
        del temp_mono_enc
        
        # print("@!@!@len ", len(x), len(x[0][0]), len(x[-1][0]))   
        try:
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\nx[0], x[1], x[2] shape is ", x[0].shape, x[1].shape, x[2].shape)
            x1 = torch.cat(x, dim=1)

        except AssertionError as error:
            for data in x:
                #print("##################inside assertion error!!")
                print(data.size())

        try:

            # print("@@@@@@@@@@@@@@@@@@@x1 ", x1.shape)
            # print("@@@@@@@@@@@@@@@@@@@rnn_hidden_states ", rnn_hidden_states.shape)

            # if(x1.shape == (1, 1536)):
            #     # print("*##*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#coming inside this\n\n")  # added by Sanjoy
            #     x1 = torch.cat((x1,x1), 0)

            # print("#@#@#@#x1.shape ", x1.shape)
            x2, rnn_hidden_states_new = self.state_encoder(x1, rnn_hidden_states, masks)
            # print("@@@@@@@@@@@@@@@@@@@x2 ", x2.shape)
            # print("@@@@@@@@@@@@@@@@@@@rnn_hidden_states new ", rnn_hidden_states_new.shape)
            #assert False
        except AssertionError as error:
            print(x1.size(), rnn_hidden_states.size(), masks.size(), x2.size(), rnn_hidden_states_new.size())

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states_new


class PassiveSepEnc(nn.Module):
    r"""Network which encodes separated bin or mono outputs
    """
    def __init__(self, convert_bin2mono=False,):
        super().__init__()

        self.passive_sep_encoder = PassiveSepEncCNN(convert_bin2mono=convert_bin2mono,)

    def forward(self, observations, mixed_audio=None):
        bottleneck_feats, lst_skip_feats = self.passive_sep_encoder(observations, mixed_audio=mixed_audio,)

        return bottleneck_feats, lst_skip_feats


class PassiveSepDec(nn.Module):
    r"""Network which decodes separated bin or mono outputs feature embeddings
    """
    def __init__(self, convert_bin2mono=False,):
        super().__init__()
        self.passive_sep_decoder = PassiveSepDecCNN(convert_bin2mono=convert_bin2mono,)

    def forward(self, bottleneck_feats, lst_skip_feats):
        return self.passive_sep_decoder(bottleneck_feats, lst_skip_feats)


class Policy(nn.Module):
    r"""
    Network for the full AAViDSS policy, including separation and action-making
    """
    def __init__(self, pol_net, dim_actions, binSep_enc, binSep_dec, bin2mono_enc, bin2mono_dec, audio_mem, ppo_cfg,):
        super().__init__()
        self.dim_actions = dim_actions

        # full policy with actor and critic
        self.pol_net = pol_net
        self.action_dist = CategoricalNet(
            self.pol_net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.pol_net.output_size)

        self.binSep_enc = binSep_enc
        self.binSep_dec = binSep_dec
        self.bin2mono_enc = bin2mono_enc
        self.bin2mono_dec = bin2mono_dec
        self.audio_mem = audio_mem

        self.ppo_cfg = ppo_cfg
        self.sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        self.poseEnc_cfg = self.sepExtMem_cfg.POSE_ENCODING

    def forward(self):
        raise NotImplementedError

    def get_binSepMasks(self, observations):
        bottleneck_feats,  lst_skip_feats = self.binSep_enc(
            observations,
        )
        return self.binSep_dec(bottleneck_feats, lst_skip_feats)

    def convert_bin2mono(self, pred_binSepMasks, mixed_audio=None):
        bottleneck_feats,  lst_skip_feats = self.bin2mono_enc(
            pred_binSepMasks, mixed_audio=mixed_audio
        )
        return self.bin2mono_dec(bottleneck_feats, lst_skip_feats)

    def get_monoFromMem(self,
                        pred_mono,
                        sepExtMem_mono,
                        sepExtMem_masks,
                        pose,
                        sepExtMem_skipFeats=None,
                        ):
        pred_mono_feats, pred_skipFeats = self.audio_mem.encode_mono(pred_mono)
        assert len(pred_skipFeats) == 1, "implemented for just 1 skip connection b/w transformer encoder and decoder"

        M, bs = sepExtMem_mono.size()[:2]

        sepExtMem_pose = sepExtMem_mono[..., -self.poseEnc_cfg.num_pose_attrs:]

        sepExtMem_mono = sepExtMem_mono[..., :-self.poseEnc_cfg.num_pose_attrs]
        sepExtMem_mono = sepExtMem_mono.contiguous().view(sepExtMem_mono.size(0),
                                                          sepExtMem_mono.size(1),
                                                          512,
                                                          32).unsqueeze(-1)
        sepExtMem_mono = sepExtMem_mono.view(-1, 512, 32, 1)
        sepExtMem_mono_feats, sepExtMem_skipFeats_new = self.audio_mem.encode_mono(sepExtMem_mono)
        sepExtMem_mono_feats = sepExtMem_mono_feats.contiguous().view(M, bs, -1)
        sepExtMem_mono_feats = torch.cat((sepExtMem_mono_feats, sepExtMem_pose), dim=-1)

        sepExtMem_skipFeats =\
            sepExtMem_skipFeats_new[0].contiguous().view(M,
                                                         bs,
                                                         sepExtMem_skipFeats_new[0].size(1),
                                                         sepExtMem_skipFeats_new[0].size(2),
                                                         sepExtMem_skipFeats_new[0].size(3))

        mono_selfAttn_outFeats, skip_feats =\
            self.audio_mem.get_selfAttn_outFeats(pose,
                                                 pred_mono_feats,
                                                 pred_skipFeats[0],
                                                 sepExtMem_mono_feats,
                                                 sepExtMem_skipFeats,
                                                 sepExtMem_masks,
                                                 )

        pred_mono_toCache =\
            pred_mono.squeeze(-1).view(pred_mono.size(0), -1)

        pred_mono_toCache = torch.cat((pred_mono_toCache, pose), dim=-1)

        pred_monoFromMem = self.audio_mem.upsample_selfAttn_outFeats(mono_selfAttn_outFeats,
                                                                     skip_feats=skip_feats,)

        """returning before attn features for caching to memory"""
        return pred_monoFromMem, pred_mono_toCache.detach(), mono_selfAttn_outFeats.detach(), pred_skipFeats[0].detach()

    def act(
        self,
        observations,
        rnn_hidden_states_pol,
        masks,
        pred_monoFromMem,
        pred_binSepMasks,
        deterministic=False,
    ):
        feats_pol, rnn_hidden_states_pol = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks.detach(),
            pred_monoFromMem.detach()
        )

        dist = self.action_dist(feats_pol)
        value = self.critic(feats_pol)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states_pol, dist.get_probs()

    def get_value(
            self,
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_monoFromMem,
            pred_binSepMasks,
    ):
        feats_pol, _ = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks.detach(),
            pred_monoFromMem.detach()
        )
        return self.critic(feats_pol)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states_pol,
        masks,
        action,
        pred_monoFromMem,
        pred_binSepMasks,
    ):
        feats_pol, rnn_hidden_states_pol = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks,
            pred_monoFromMem,
        )

        dist = self.action_dist(feats_pol)
        value = self.critic(feats_pol)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hidden_states_pol


class AAViDSSPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        extra_rgb,
        extra_depth,
        ppo_cfg,
    ):

        pol_net = PolicyNet(
            observation_space=observation_space,
            hidden_size=ppo_cfg.hidden_size,
            extra_rgb=extra_rgb,
            extra_depth=extra_depth,
            use_location_in_policy=ppo_cfg.use_location_in_policy,
            use_predictedLocation_in_policy=ppo_cfg.use_predictedLocation_in_policy,
            locationPredictor_encoderType=ppo_cfg.locationPredictor_encoderType,
            use_predictedLocationWoFeats_in_policy=ppo_cfg.use_predictedLocationWoFeats_in_policy, 
        )

        binSep_enc = PassiveSepEnc()
        binSep_dec = PassiveSepDec()

        bin2mono_enc = PassiveSepEnc(
            convert_bin2mono=True,
        )
        bin2mono_dec = PassiveSepDec(
            convert_bin2mono=True,
        )

        audio_mem = AudioMem(
            ppo_cfg
        )

        super().__init__(
            pol_net,
            action_space.n,
            binSep_enc,
            binSep_dec,
            bin2mono_enc,
            bin2mono_dec,
            audio_mem,
            ppo_cfg,
        )