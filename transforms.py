import random
import torch
import logging
import itertools
import numpy as np
import logging
import sys
import torch.nn as nn
import torchaudio
import torchaudio.transforms
import torchaudio.compliance.kaldi
import math

def _complex_mul(a, b):
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def convolve(signal, kernel, mode='full'):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.
    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    :rtype torch.Tensor:
    """
    m = signal.size(-1)
    n = kernel.size(-1)
    if mode == 'full':
        truncate = m + n - 1
    elif mode == 'valid':
        truncate = max(m, n) - min(m, n) + 1
    elif mode == 'same':
        truncate = max(m, n)
    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up to next power of 2 for cheaper fft.
    fast_ftt_size = 2 ** math.ceil(math.log2(padded_size))
    f_signal = torch.rfft(torch.nn.functional.pad(signal, (0, fast_ftt_size - m)), 1, onesided=False)
    f_kernel = torch.rfft(torch.nn.functional.pad(kernel, (0, fast_ftt_size - n)), 1, onesided=False)
    f_result = _complex_mul(f_signal, f_kernel)
    result = torch.irfft(f_result, 1, onesided=False)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx: start_idx + truncate]

class Reverberate(nn.Module):

    def __init__(self, rir_list_filename, sample_rate=16000, retain_power=True):
        super(Reverberate, self).__init__()
        self.rirs = []
        self.retain_power = retain_power
        for l in open(rir_list_filename):
            wav, rev_sample_rate = torchaudio.load(l.strip(), normalization=1 << 31)
            if rev_sample_rate != sample_rate:
                wav = torchaudio.compliance.kaldi.resample_waveform(wav, rev_sample_rate, sample_rate) 
            self.rirs.append(wav[0])

    def forward(self, x):
        result = convolve(x, random.sample(self.rirs, 1)[0].to(x.device))[..., :x.shape[-1]]
        if self.retain_power:
            power_before = torch.dot(x, x) / len(x)
            power_after = torch.dot(result, result) / len(result)
            #breakpoint()
            result *= (power_before / power_after).sqrt()
            result = torch.clamp(result, -1.0, 1.0)
            
        return result


class AddNoise(nn.Module):

    def __init__(self, noise_list_filename, min_lambda=0.5, max_lambda=0.8, sample_rate=16000):
        super(AddNoise, self).__init__()
        self.noises = []
        self.min_lambda = min_lambda
        self.max = max_lambda
        for l in open(noise_list_filename):
            wav, noise_sample_rate = torchaudio.load(l.strip(), normalization=1 << 31)
            if noise_sample_rate != sample_rate:
                wav = torchaudio.compliance.kaldi.resample_waveform(wav, noise_sample_rate, sample_rate) 
            self.noises.append(wav[0])

    def forward(self, x):
        l = random.uniform(self.min_lambda, self.min_lambda)
        noise = random.sample(self.noises, 1)[0]
        #breakpoint()
        if len(noise) > len(x):
            noise_start = random.randint(0, len(noise) - len(x))
            noise_end = noise_start + len(x)
            result = (1 - l) * x + l * noise[noise_start : noise_end].to(x.device)
            return torch.clamp(result, -1.0, 1.0)
        else:
            x_start = random.randint(0, len(x) - len(noise))
            x_end = x_start + len(noise)
            x *= 1 - l
            x[x_start:x_end] += l * noise.to(x.device)
            x = torch.clamp(x, -1.0, 1.0)
            return x


class SpeedPerturbation(nn.Module):
    def __init__(self, speeds=[0.9, 1.1]):
        super(SpeedPerturbation, self).__init__()
        self.speeds = speeds
        
    def forward(self, x):
        speed = random.sample(self.speeds, 1)[0]
        result = torchaudio.compliance.kaldi.resample_waveform(x.reshape(-1, x.shape[-1]), 16000, 16000 * speed, lowpass_filter_width=6)
        if len(x.shape) == 1:
            result = result.squeeze(0)
        return result

        

class WhiteNoise(nn.Module):
    def __init__(self, noise_scl=0.01):
        super(WhiteNoise, self).__init__()
        self.noise_scl = noise_scl

    def forward(self, x):
        noise = torch.randn(x.shape, device=x.device) * self.noise_scl 
        return x + noise        
        

class Noop(nn.Module):
    def __init__(self):
        super(Noop, self).__init__()

    def forward(self, x):
        return x

class FreqMask(nn.Module):
    def __init__(self, max_masked_freqs=5, num_masks=1, replace_with_zero=False):
        super(FreqMask, self).__init__()
        self.replace_with_zero = replace_with_zero
        self.num_masks = num_masks
        self.max_masked_freqs = max_masked_freqs

    def forward(self, x):
        assert len(x.shape) == 2
        
        num_freqs = x.shape[0]
        for i in range(0, self.num_masks):        
            f = random.randrange(1, self.max_masked_freqs + 1)
            f_zero = random.randrange(0, num_freqs - f)

            mask_end = f_zero + f            
            if (self.replace_with_zero): 
                x[f_zero:mask_end] = 0
            else: 
                x[f_zero:mask_end] = x.mean()
        return x

def augment_and_mix(transforms, wav):
    severity=3
    width=2
    depth=-1
    alpha=1.

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = torch.zeros_like(wav)
    wav_aug = wav.clone().detach()
    for i in range(width):
        wav_aug.copy_(wav)
        depth = depth if depth > 0 else np.random.randint(1, 3)
        for _ in range(depth):
            op = np.random.choice(transforms)
            wav_aug = op(wav_aug)
            # Preprocessing commutes since all coefficients are convex
        #breakpoint()
        mix += ws[i] * wav_aug

    mixed = (1 - m) * wav + m * mix
    return mixed


if __name__ == "__main__":
    transforms = [
        ("rvb", Reverberate(rir_list_filename="test/real_and_sim_rirs.wavs.txt")),
        ("add_noise", AddNoise(noise_list_filename="test/musan.100.wavs.txt")),
        ("white_noise", WhiteNoise())        
    ]

    for name, t in transforms:
        wav, sample_rate = torchaudio.load("test/source.wav", normalization=1 << 31)
        wav = wav[0]

        transformed = t(wav)
        #breakpoint()
        torchaudio.save(f"test/source_{name}.wav", transformed.unsqueeze(0), sample_rate)

    for i in range(5):
        wav, sample_rate = torchaudio.load("test/source.wav", normalization=1 << 31)
        wav = wav[0]
        mixed = augment_and_mix([t for name, t in transforms], wav)
        torchaudio.save(f"test/mixed_{i}.wav", mixed.unsqueeze(0), sample_rate)
