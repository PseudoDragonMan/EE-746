import torch
import snntorch as snn
import numpy as np
import torchvision.transforms as transforms
import tonic
from torch.utils.data import DataLoader

class LSM(torch.nn.Module):
    def __init__(self, N, in_sz, Win, Wlsm, alpha=0.9, beta=0.9, th=20, tau_pos=20, tau_neg=20):
        super(LSM, self).__init__()
        self.fc1 = torch.nn.Linear(in_sz, N)
        self.fc1.weight = torch.nn.Parameter(torch.from_numpy(Win))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.Wlsm = torch.nn.Parameter(torch.from_numpy(Wlsm))
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def apply_stdp(self, pre_spike_time, post_spike_time, weights):
        dt = pre_spike_time - post_spike_time
        dt_pos = np.exp(-dt / self.tau_pos)
        dt_neg = np.exp(dt / self.tau_neg)

        # Apply STDP update rule
        weights += (dt_pos - dt_neg) * 0.001  # Adjust the learning rate as needed
        weights = np.clip(weights, -1, 1)  # Clip weights to a valid range
        return weights

    def forward(self, x):
        num_steps = x.size(0)
        spk, syn, mem = self.lsm.init_rsynaptic()
        spk_rec = []
        for step in range(num_steps):
            curr = self.fc1(x[step])
            spk, syn, mem = self.lsm(curr, spk, syn, mem)

            # Apply STDP to the recurrent weights
            if step > 0:
                post_spikes = torch.where(spk[0] == 1)[0].numpy()
                pre_spikes = torch.where(spk_rec[step - 1] == 1)[0].numpy()

                for pre_idx in pre_spikes:
                    for post_idx in post_spikes:
                        self.Wlsm[post_idx, pre_idx] = torch.from_numpy(self.apply_stdp(
                            step, step - 1, self.Wlsm[post_idx, pre_idx].numpy()
                        ))

            spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out
