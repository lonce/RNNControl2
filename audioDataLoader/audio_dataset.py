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

from pathlib import Path
import pandas as pd

@dataclass
class AudioDatasetConfig:
    data_dir: str                                  # Path to directory with .wav files
    sequence_length: int                           # Number of samples per training sequence
    parameter_specs: Dict[str, Tuple[float, float]]  # {'param': (min, max), ...}
    data_manifest: str = ""                        # Path to xlsx file with datafile list and parameters
    add_noise: bool = False                        # Whether to add white noise
    noise_weight: float = 0.1                      # Weight for uniform noise injection
    encode: bool = False                           # False - float[-1,1], True - [0,1] compressed mulaw
    codebook_size: int = 256
    
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

        # Target is created from the original, clean audio signal
        target_seq = mu_law_encode(chunk[1:], quantization_channels=self.config.codebook_size).long()

        # Input sequence is prepared separately
        input_chunk = chunk[:-1]

        if self.config.add_noise:
            input_chunk = self._add_noise(input_chunk, self.config.noise_weight)

        if self.config.encode:
            encoded_input = mu_law_encode(input_chunk, quantization_channels=self.config.codebook_size)
            input_seq = encoded_input.float() / (self.config.codebook_size-1.0)
        else:
            input_seq = input_chunk
        
        cond_params = norm_params.unsqueeze(0).expand(input_seq.shape[0], -1)
        
        # Combine audio and conditioning parameters into a single input tensor
        input_tensor = torch.cat([input_seq.unsqueeze(-1), cond_params], dim=-1)
        
        # CHANGE: Add extra dimension for n_q compatibility
        # Shape changes from (seq_len,) to (seq_len, 1)
        return input_tensor, target_seq.unsqueeze(-1).unsqueeze(-1)

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

###################################################################################################

class MuLawAudioDatasetFromManifest(Dataset):
    """
    Same behavior as MuLawAudioDataset2, except:
      - Raw (pre-normalization) parameter values are read from an XLSX manifest.
      - We use ONLY the parameters listed in config.parameter_specs.
        parameter_specs: Ordered mapping of {param_name: (min_val, max_val)}
      - The conditioning vector order matches the iteration order of parameter_specs.

    Manifest requirements:
      - Column 'filename' listing the wav file name (not path).
      - One column per parameter present in config.parameter_specs.

    Other config fields (data_dir, sequence_length, encode, codebook_size, add_noise, etc.)
    are used exactly as in your existing dataset.
    """

    def __init__(self, config):
        self.config = config
        self.sequence_length = config.sequence_length
        self.parameter_specs = config.parameter_specs  # dict-like, order matters
        self.data = []          # list of (waveform_1D, norm_params_1D)
        self.sequence_map = []  # list of (file_index, start_pos)

        # --- Load manifest ---
        manifest_path = Path(config.data_manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_excel(manifest_path)
        if "filename" not in df.columns:
            raise ValueError("Manifest must contain a 'filename' column")

        # Ensure filenames are clean strings
        df["filename"] = df["filename"].astype(str).str.strip()

        print(f'self.parameter_specs = {self.parameter_specs}')

        # Verify required parameter columns exist
        required_params = list(self.parameter_specs.keys())
        missing = [p for p in required_params if p not in df.columns]
        if missing:
            raise ValueError(
                "Manifest is missing required parameter columns: "
                + ", ".join(missing)
            )

        # Build a lookup: filename -> dict(param_name -> raw_value)
        param_lookup = {}
        for _, row in df.iterrows():
            fname = row["filename"]
            values = {}
            for p in required_params:
                try:
                    values[p] = float(row[p])
                except Exception as e:
                    raise ValueError(f"Bad value for '{p}' in '{fname}': {e}")
            param_lookup[fname] = values

        # --- Preload waveforms and normalized params (use manifest rows as source of truth) ---
        data_dir = Path(config.data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        loaded = 0
        for _, row in df.iterrows():
            fname = row["filename"]
            wav_path = data_dir / fname
            if not wav_path.suffix.lower().endswith(".wav"):
                # ignore non-wav entries silently
                continue
            if not wav_path.is_file():
                # You can switch to 'raise' if you want strict behavior
                # raise FileNotFoundError(f"WAV not found: {wav_path}")
                continue

            # Load mono waveform, clamp to [-1,1]
            waveform, sr = torchaudio.load(str(wav_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.squeeze(0)
            waveform = torch.clamp(waveform, -1.0, 1.0)

            # Normalize only the requested parameters, in the order of parameter_specs
            norm_params = self._normalize_params(param_lookup[fname])
            self.data.append((waveform, norm_params))
            loaded += 1

        if loaded == 0:
            print(f"[MuLawAudioDatasetFromManifest] Warning: no WAVs loaded from {data_dir}")

        # --- Build sequence map exactly like MuLawAudioDataset2 ---
        for file_idx, (waveform, _) in enumerate(self.data):
            nseq = len(waveform) - self.sequence_length
            if nseq <= 0:
                continue
            for start in range(nseq):
                self.sequence_map.append((file_idx, start))

        self.indexLen = len(self.sequence_map)

    def __len__(self):
        return self.indexLen

    def __getitem__(self, idx):
        file_idx, start_pos = self.sequence_map[idx]
        waveform, norm_params = self.data[file_idx]

        end_pos = start_pos + self.sequence_length + 1
        chunk = waveform[start_pos:end_pos]  # length = seq_len + 1

        # Target (from clean original), mu-law encoded to int class labels
        target_seq = mu_law_encode(
            chunk[1:], quantization_channels=self.config.codebook_size
        ).long()

        # Input (optionally noisy), optionally mu-law-encoded and scaled to [0,1]
        input_chunk = chunk[:-1]
        if getattr(self.config, "add_noise", False):
            input_chunk = self._add_noise(input_chunk, getattr(self.config, "noise_weight", 0.0))

        if getattr(self.config, "encode", True):
            encoded_input = mu_law_encode(
                input_chunk, quantization_channels=self.config.codebook_size
            )
            input_seq = encoded_input.float() / (self.config.codebook_size - 1.0)
        else:
            input_seq = input_chunk

        # Repeat normalized params across time and concatenate as last dimension
        cond_params = norm_params.unsqueeze(0).expand(input_seq.shape[0], -1)
        input_tensor = torch.cat([input_seq.unsqueeze(-1), cond_params], dim=-1)

        # CHANGE: Add extra dimension for n_q compatibility  
        # Shape changes from (seq_len,) to (seq_len, 1)
        return input_tensor, target_seq.unsqueeze(-1).unsqueeze(-1)

    # ---------- Helpers ----------

    def _add_noise(self, sample, weight):
        if weight == 0:
            return sample
        noise = torch.from_numpy(
            np.random.uniform(sample.min().item(), sample.max().item(), size=sample.shape[0])
        ).float()
        return sample + weight * noise

    def _normalize_params(self, raw_param_dict):
        """
        Apply min-max normalization per parameter using parameter_specs:
            norm = (val - vmin) / (vmax - vmin)   (clamped to [0,1])
        Order follows parameter_specs iteration order.
        """
        out = []
        for key, (vmin, vmax) in self.parameter_specs.items():
            if key not in raw_param_dict:
                raise KeyError(f"Parameter '{key}' missing from manifest row")
            raw_val = float(raw_param_dict[key])
            denom = (vmax - vmin)
            if denom == 0:
                norm = 0.0
            else:
                norm = (raw_val - vmin) / denom
            # Optional clamp for safety
            norm = max(0.0, min(1.0, norm))
            out.append(norm)
        return torch.tensor(out, dtype=torch.float32)

    def rand_sample(self, idx=None):
        if idx is None:
            idx = random.randint(0, self.indexLen - 1)
        file_idx, start_pos = self.sequence_map[idx]
        waveform, pvect = self.data[file_idx]
        end_pos = start_pos + self.sequence_length + 1
        return waveform[start_pos:end_pos], pvect