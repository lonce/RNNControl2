import numpy as np
import torch
import torchaudio

# torch.from_numpy()
# .numpy
USETORCHAUDO=False

if USETORCHAUDO :
    def mu_law_encode(audio_float, quantization_channels=256) :
        if isinstance(audio_float, np.ndarray):
            return torchaudio.functional.mu_law_encoding(torch.from_numpy(audio_float), quantization_channels=256).numpy()
        else:
            return torchaudio.functional.mu_law_encoding(audio_float, quantization_channels=256)

    def  mu_law_decode(audio_mulaw, quantization_channels=256):
        if isinstance(audio_mulaw, np.ndarray):
            return torchaudio.functional.mu_law_decoding(torch.from_numpy(audio_mulaw), quantization_channels=256).numpy()
        else : 
            return torchaudio.functional.mu_law_decoding(audio_mulaw, quantization_channels=256)

else : 

    def mu_law_encode(audio, quantization_channels=256):
        """
        Encode waveform using mu-law companding and map to integer indices in [0, quantization_channels - 1].
    
        Parameters:
            audio: np.ndarray or torch.Tensor, must be in [-1, 1]
            quantization_channels: Number of quantization channels (typically 256)
    
        Returns:
            Encoded integer values (np.int32 or torch.long)
        """
        mu = quantization_channels - 1
    
        if isinstance(audio, np.ndarray):
            safe_audio = np.clip(audio, -1.0, 1.0)
            magnitude = np.log1p(mu * np.abs(safe_audio)) / np.log1p(mu)
            signal = np.sign(safe_audio) * magnitude
            return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)
    
        elif isinstance(audio, torch.Tensor):
            safe_audio = torch.clamp(audio, -1.0, 1.0)
            magnitude = torch.log1p(mu * torch.abs(safe_audio)) / np.log1p(mu)
            signal = torch.sign(safe_audio) * magnitude
            return ((signal + 1) / 2 * mu + 0.5).long()
    
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")
    
    
        
    def mu_law_decode(encoded, quantization_channels=256):
        """
        Decode integer mu-law encoded values in [0, quantization_channels - 1] back to waveform in [-1, 1].
    
        Parameters:
            encoded: np.ndarray or torch.Tensor of integer values
            quantization_channels: Number of quantization channels (typically 256)
    
        Returns:
            Reconstructed waveform in [-1, 1] (same type as input)
        """
        mu = quantization_channels - 1
    
        if isinstance(encoded, np.ndarray):
            signal = 2 * (encoded.astype(np.float32) / mu) - 1
            magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
            return np.sign(signal) * magnitude
    
        elif isinstance(encoded, torch.Tensor):
            signal = 2 * (encoded.float() / mu) - 1
            magnitude = (1 / mu) * ((1 + mu) ** torch.abs(signal) - 1)
            return torch.sign(signal) * magnitude
    
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")
