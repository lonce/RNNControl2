import os
import re
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Tuple, List
import random
from .mulaw import mu_law_encode, mu_law_decode


@dataclass
class AudioDatasetConfig:
    data_dir: str                                  # Path to directory with .wav files
    sequence_length: int                           # Number of samples per training sequence
    parameter_specs: Dict[str, Tuple[float, float]]  # {'param': (min, max), ...}
    add_noise: bool = False                        # Whether to add white noise
    noise_weight: float = 0.1                      # Weight for uniform noise injection
    encode: bool = False                           # False - float[-1,1], True - [0,1] compressed mulaw
    
####################################################################################################################
class MuLawAudioDataset2(Dataset):
    def __init__(self, config: AudioDatasetConfig):
        self.config = config
        self.sequence_length = config.sequence_length
        self.parameter_specs = config.parameter_specs
        self.data = []  # Holds (waveform, norm_params) tuples
        self.sequence_map = []  # Holds (file_index, start_pos)

        # Preload all waveforms and conditioning params
        for filename in sorted(os.listdir(config.data_dir)):
            if filename.endswith(".wav"):
                filepath = os.path.join(config.data_dir, filename)
                waveform, sr = torchaudio.load(filepath)
                
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0) # Mono

                waveform = waveform.squeeze(0)

                

                waveform = torch.clamp(waveform, -1.0, 1.0)

                norm_params = self._parse_and_normalize_params(filename)
                if norm_params is not None:
                    self.data.append((waveform, norm_params))

        # Create a map of all possible sequences
        for file_idx, (waveform, _) in enumerate(self.data):
            num_sequences = len(waveform) - self.sequence_length
            for i in range(num_sequences):
                self.sequence_map.append((file_idx, i))
        self.indexLen=len(self.sequence_map)

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        file_idx, start_pos = self.sequence_map[idx]
        waveform, norm_params = self.data[file_idx]

        end_pos = start_pos + self.sequence_length + 1
        chunk = waveform[start_pos:end_pos]

        if self.config.add_noise:
            chunk = self._add_noise(chunk, self.config.noise_weight)

        encoded = mu_law_encode(chunk, quantization_channels=256)
        
        if self.config.encode:
            input_seq = encoded[:-1].float() / 255.0
        else:
            input_seq=chunk
        target_seq = encoded[1:].long()

        cond_params = norm_params.unsqueeze(0).expand(input_seq.shape[0], -1)
        
        # Combine audio and conditioning parameters into a single input tensor
        input_tensor = torch.cat([input_seq.unsqueeze(-1), cond_params], dim=-1)
        
        return input_tensor, target_seq.unsqueeze(-1)

    def _add_noise(self, sample, weight):
        noise = torch.from_numpy(np.random.uniform(sample.min(), sample.max(), size=len(sample))).float()
        return sample + weight * noise

    def _parse_and_normalize_params(self, filename):
        result = []
        for key, (vmin, vmax) in self.parameter_specs.items():
            pattern = rf"_{key}(-?\d+\.\d+)"
            match = re.search(pattern, filename)
            if not match:
                return None
            raw_val = float(match.group(1))
            norm_val = (raw_val - vmin) / (vmax - vmin)
            result.append(norm_val)
        return torch.tensor(result, dtype=torch.float32)

    def rand_sample(self, idx=None):
        """
        Return a randomly chosen audio segment (mu-law decoded) from the dataset.
        If n is specified, truncate or zero-pad to exactly n samples.
        """

        if idx is None:
            idx = random.randint(0, self.indexLen - 1)

        print(f"rand_sample: idx={idx}")
        file_idx, start_pos = self.sequence_map[idx]
        
        print(f"rand_sample: file_idx={file_idx}")
        
        waveform, _ = self.data[file_idx]
        end_pos = start_pos + self.sequence_length + 1

        print(f"rand_sample: start_pos={start_pos}, end_pos={end_pos}")
        
        chunk = waveform[start_pos:end_pos]
        return chunk
##############################################################################################
 
def generate_noisy_sine_warmup(steps: int, sample_rate: int = 16000, amplitude: float = 0.9, freq: float = 440.0, noise_level: float = 0.05) -> torch.Tensor:
    """
    Generate a noisy 440Hz sine wave, mu-law encode it, and return tokens.

    Args:
        steps (int): Number of time steps to generate.
        sample_rate (int): Sample rate in Hz.
        amplitude (float): Peak amplitude of the sine wave, in [0,1].
        freq (float): Frequency of sine wave in Hz.
        noise_level (float): Std dev of white noise to add (0.0 for pure tone).

    Returns:
        torch.Tensor: Mu-law encoded token sequence, shape [steps], dtype uint8.
    """
    duration_sec = steps / sample_rate
    t = np.linspace(0, duration_sec, steps, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    noisy_wave = sine_wave + np.random.normal(0, noise_level, sine_wave.shape)
    noisy_wave = np.clip(noisy_wave, -1.0, 1.0)
    waveform = torch.tensor(noisy_wave, dtype=torch.float32)

    encoded = mu_law_encode(waveform.unsqueeze(0), quantization_channels=256).squeeze(0)
    return encoded

        