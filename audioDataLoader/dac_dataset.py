import os
import re
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import random
from .dac_encode import flatten_dac_codes
import dac
from dac import DACFile

@dataclass
class AudioDatasetConfig:
    data_dir: str = 'data/dac_encoded_audio'                                # Path to directory with .wav files
    sequence_length: int = 256                           # Number of samples per training sequence
    parameter_specs: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"instID": (1, 2), "a": (0,1), "p": (64.0, 76.0)})  # {'param': (min, max), ...}
    add_noise: bool = False                        # Whether to add white noise
    noise_weight: float = 0.1                      # Weight for uniform noise injection
    encode: bool = False                           # False - float[-1,1], True - [0,1] compressed mulaw
    numtokens: int = 256
    
class DACDataset(Dataset):
    def __init__(self, config: AudioDatasetConfig):
        self.config = config
        self.sequence_length = config.sequence_length
        self.parameter_specs = config.parameter_specs
        self.data = []  # Holds (waveform, norm_params) tuples
        self.sequence_map = []  # Holds (file_index, start_pos)
        model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(model_path)
        
        # Preload all waveforms and conditioning params
        for filename in sorted(os.listdir(config.data_dir)):
            if filename.endswith(".dac"):
                filepath = os.path.join(config.data_dir, filename)
                
                codes = DACFile.load(filepath).codes
                
                flattened_codes = flatten_dac_codes(codes)

                norm_params = self._parse_and_normalize_params(filename)
                
                if norm_params is not None:
                    self.data.append((flattened_codes, norm_params))
                    
        # Create a map of all possible sequences
        for file_idx, (flattened_codes, _) in enumerate(self.data):
            num_sequences = len(flattened_codes) - self.sequence_length
            for i in range(num_sequences):
                self.sequence_map.append((file_idx, i))
        self.indexLen=len(self.sequence_map)
     
    def __len__(self):
        return len(self.sequence_map)  
                    
    def __getitem__(self, idx):
        file_idx, start_pos = self.sequence_map[idx]
        flattened_codes, norm_params = self.data[file_idx]

        end_pos = start_pos + self.sequence_length + 1
        chunk = flattened_codes[start_pos:end_pos]

        # Target is created from the original, clean audio signal
        target_seq = chunk[1:]

        # Input sequence is prepared separately
        input_chunk = chunk[:-1]

        if self.config.add_noise:
            input_chunk = self._add_noise(input_chunk, self.config.noise_weight)

        """
        if self.config.encode:
            encoded_input = mu_law_encode(input_chunk, quantization_channels=self.config.numtokens)
            input_seq = encoded_input.float() / (self.config.numtokens-1.0)
        else:
            input_seq = input_chunk
        """
        input_seq = input_chunk
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
        
if __name__ == "__main__":
    config = AudioDatasetConfig()
    dataset = DACDataset(config)
    
    print(dataset[0])
        
