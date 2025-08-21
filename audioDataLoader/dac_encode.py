import dac 
import torch
from audiotools import AudioSignal
from tqdm import tqdm
import os

def flatten_dac_codes(
	codes: torch.Tensor,

) -> torch.Tensor:
	"""
	Flatten to 1D with time-major ordering:
	for t in [0..T-1], emit all channels at time t (then t+1, ...).
    """
	x = codes

	# Normalize to shape (C, T)
	if x.ndim == 3:  # (B, C, T)
		x = x[0]       
	# Time-major flatten: (C, T) -> (T, C) -> 1D
	return x.transpose(0, 1).reshape(-1)

def encode_audio(signal: AudioSignal, model: dac.DAC) -> torch.Tensor:
	signal = signal.ffmpeg_resample(44100)
	x = model.compress(signal)
	return x

def encode_dataset(data: str, destination_dir: str):
    model_path = dac.utils.download(model_type="44khz")  # Or "24khz" / "16khz"
    os.makedirs(destination_dir, exist_ok=True)
    model = dac.DAC.load(model_path)
    audio_data_path = data
    print(f'DAC encoding dataset from {audio_data_path} to {destination_dir}')
    for file in tqdm(os.listdir(audio_data_path)):
        if file.endswith('.wav'):
            signal = AudioSignal(os.path.join(audio_data_path, file))
            x = encode_audio(signal, model)
            x.save(os.path.join(destination_dir, f'{file.strip(".wav")}.dac'))


if __name__ == "__main__":
    encode_dataset('data/nsynth.64.76_sm', 'data/dac_encoded_audio')
    










