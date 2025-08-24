
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import soundfile as sf

# Local imports from your project structure
from model.gru_audio_model import RNN, GRUModelConfig
from audioDataLoader.mulaw import mu_law_encode, mu_law_decode

def run_inference(model, cond_seq, warmup_sequence, top_n=3, temperature=1.0):
    """
    Generates audio sequence based on a conditioning sequence.

    Args:
        model: The trained RNN model.
        cond_seq (torch.Tensor): The sequence of conditioning parameters. Shape: (seq_len, num_cond_params).
        warmup_sequence (torch.Tensor): A raw audio sequence to warm up the model's hidden state. Shape: (warmup_len,).
        top_n (int): The number of top predictions to sample from.
        temperature (float): Controls the randomness of predictions. Higher is more random.

    Returns:
        np.array: The generated audio waveform.
    """
    codebook_size=model.codebook_size
    device = next(model.parameters()).device
    print(f"Starting inference... with codebook_size={codebook_size}")

    # --- 1. Warm-up Phase --- 
    #print("Warming up model hidden state...")
    warmup_encoded = mu_law_encode(warmup_sequence, quantization_channels=codebook_size)
    warmup_input_audio = (warmup_encoded.float() / (codebook_size-1.0)).to(device)

    first_cond_vec = cond_seq[0].unsqueeze(0).repeat(len(warmup_input_audio), 1).to(device)
    warmup_full_input = torch.cat([warmup_input_audio.unsqueeze(-1), first_cond_vec], dim=-1)

    hidden = model.init_hidden(batch_size=1)
    for i in range(len(warmup_full_input)):
        _, hidden = model(warmup_full_input[i].unsqueeze(0), hidden, batch_size=1)

    next_input_audio = warmup_input_audio[-1].unsqueeze(0)

    # --- 2. Generation Phase --- 
    print(f"Generating {len(cond_seq)} audio samples... using codebook_size={codebook_size}")
    generated_samples = []
    with torch.no_grad():
        for i in range(len(cond_seq)):
            current_cond_vec = cond_seq[i].unsqueeze(0).to(device)
            next_input_full = torch.cat([next_input_audio.unsqueeze(-1), current_cond_vec], dim=-1)

            logits, hidden = model(next_input_full, hidden, batch_size=1)

            for j in range(model.n_q): #actually always 1 for the mulaw audio
                logits[j] = logits[j].div(temperature).squeeze()
                top_n_logits, top_n_indices = torch.topk(logits[j], top_n)
                top_n_probs = F.softmax(top_n_logits, dim=-1)
                sampled_relative_idx = torch.multinomial(top_n_probs, 1).squeeze()
    
                #manage possible NaN with some debugging statement
                try: 
                    sampled_mu_law_index = top_n_indices[sampled_relative_idx]
                except Exception as e:
                    print(f"Sampling error: {e}")
                    
                    #and more detailed info: 
                    print(f"top_n_logits stats: min={top_n_logits.min()}, max={top_n_logits.max()}, has_nan={torch.isnan(top_n_logits).any()}")
                    print(f"top_n_probs stats: min={top_n_probs.min()}, max={top_n_probs.max()}, has_nan={torch.isnan(top_n_probs).any()}")
                    print(f"sampled_relative_idx: {sampled_relative_idx}, type: {type(sampled_relative_idx)}")
                    print(f"top_n_indices shape: {top_n_indices.shape}, sampled_relative_idx shape: {sampled_relative_idx.shape}")

            

            new_audio_sample = mu_law_decode(sampled_mu_law_index, quantization_channels=codebook_size)
            generated_samples.append(new_audio_sample.item())

            next_input_audio = (mu_law_encode(new_audio_sample.unsqueeze(0), codebook_size).float() / (codebook_size-1.0)).to(device)

    print("Inference complete.")
    return np.array(generated_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio from a trained GRU model.')
    parser.add_argument('--run_directory', type=str, required=True, help='Path to the directory of the saved run.')
    parser.add_argument('--top_n', type=int, default=5, help='Sample from the top N most likely outputs.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Controls the randomness of predictions.')
    parser.add_argument('--length_seconds', type=float, default=2.0, help='Length of the audio to generate in seconds.')
    parser.add_argument('--output_wav_path', type=str, default='generated_audio.wav', help='Path to save the output WAV file.')
    parser.add_argument('--output_plot_path', type=str, default='generated_waveform.png', help='Path to save the output plot.')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config_path = os.path.join(args.run_directory, "config.pt")
    checkpoint_path = os.path.join(args.run_directory, "checkpoints", "last_checkpoint.pt")

    assert os.path.exists(args.run_directory), f"Run directory not found: {args.run_directory}"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    assert os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    saved_configs = torch.load(config_path)
    model_config = saved_configs["model_config"]

    model = RNN(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model successfully loaded from checkpoint.")

    sample_rate = 16000
    generation_length = int(args.length_seconds * sample_rate)

    num_cond_params = model_config.cond_size
    cond_seq = torch.zeros(generation_length, num_cond_params)
    cond_seq[:, 0] = 0.0
    cond_seq[:, 1] = 0.8
    cond_seq[:, 2] = torch.linspace(0, 1, generation_length)

    warmup_len = 512
    t = torch.linspace(0., 1., warmup_len)
    warmup_sequence = torch.sin(2 * np.pi * 220.0 * t)

    generated_audio = run_inference(
        model=model,
        cond_seq=cond_seq,
        warmup_sequence=warmup_sequence,
        top_n=args.top_n,
        temperature=args.temperature
    )

    print(f"Saving generated audio to {args.output_wav_path}")
    sf.write(args.output_wav_path, generated_audio, sample_rate)

    print(f"Saving waveform plot to {args.output_plot_path}")
    plt.figure(figsize=(20, 5))
    plt.plot(generated_audio)
    plt.title("Generated Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.savefig(args.output_plot_path)
    plt.close()

    print("Done.")
