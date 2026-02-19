import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

def split_audio(audio_path, output_dir, segment_duration=3, sr=22050):
    """
    Split a single audio file into segments of specified duration.
    
    Parameters:
    - audio_path: path to the audio file
    - output_dir: directory to save segments
    - segment_duration: duration of each segment in seconds
    - sr: sampling rate
    
    Returns:
    - List of paths to saved segments (empty list if file is corrupted)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"    ⚠️ Cannot load audio file (corrupted or unsupported format): {e}")
        return []
    
    # Calculate samples per segment
    samples_per_segment = int(segment_duration * sr)
    
    # Calculate number of segments
    total_samples = len(y)
    num_segments = total_samples // samples_per_segment
    
    segment_paths = []
    
    # Split and save segments
    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = y[start_sample:end_sample]
        
        # Create output filename
        base_name = Path(audio_path).stem
        segment_filename = f"{base_name}_segment_{i:02d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        
        # Save segment
        sf.write(segment_path, segment, sr)
        segment_paths.append(segment_path)
    
    return segment_paths

def audio_to_spectrogram_gpu(audio_path, output_path, sr=22050, n_mels=128, fmax=8000):
    """
    GPU-accelerated version: Convert audio file to mel spectrogram image using PyTorch.
    
    Parameters:
    - audio_path: path to audio file
    - output_path: path to save spectrogram image
    - sr: sampling rate
    - n_mels: number of mel bands
    - fmax: maximum frequency
    """
    # Load audio with torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != sr:
        resampler = T.Resample(sample_rate, sr).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Create mel spectrogram transform
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        f_max=fmax,
    ).to(device)
    
    # Generate mel spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to dB scale
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    
    # Move back to CPU for plotting
    mel_spec_db_np = mel_spec_db.squeeze().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db_np, sr=sr, fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def process_gtzan_dataset(input_dir, output_audio_dir, output_spec_dir, 
                          segment_duration=3, sr=22050, save_segments=True, use_gpu=True):
    """
    Process entire GTZAN dataset: split audio and create spectrograms.
    
    Parameters:
    - input_dir: directory containing GTZAN dataset (with genre subdirectories)
    - output_audio_dir: directory to save audio segments
    - output_spec_dir: directory to save spectrogram images
    - segment_duration: duration of each segment in seconds
    - sr: sampling rate
    - save_segments: whether to save audio segments (False to only create spectrograms)
    - use_gpu: whether to use GPU acceleration for spectrograms
    """
    # Create output directories
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_spec_dir, exist_ok=True)
    
    # Track overall progress
    start_time = __import__('time').time()
    total_genres_processed = 0
    
    # Process each genre
    input_path = Path(input_dir)
    
    # Check if input_dir has genre subdirectories or direct audio files
    audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.au'))
    
    if audio_files:
        # Files are directly in input_dir
        process_genre(input_dir, None, output_audio_dir, output_spec_dir, 
                     segment_duration, sr, save_segments, use_gpu)
        total_genres_processed = 1
    else:
        # Files are in genre subdirectories
        genre_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        
        print(f"Found {len(genre_dirs)} genres to process: {[g.name for g in genre_dirs]}\n")
        
        for idx, genre_dir in enumerate(genre_dirs, 1):
            genre = genre_dir.name
            print(f"[{idx}/{len(genre_dirs)}] Processing genre: {genre}")
            print("=" * 60)
            
            process_genre(str(genre_dir), genre, output_audio_dir, output_spec_dir, 
                         segment_duration, sr, save_segments, use_gpu)
            total_genres_processed += 1
            print()
    
    # Final summary
    elapsed_time = __import__('time').time() - start_time
    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total genres processed: {total_genres_processed}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Output directories:")
    print(f"  - Audio segments: {output_audio_dir}")
    print(f"  - Spectrograms: {output_spec_dir}")
    print("=" * 60)

def process_genre(genre_path, genre_name, output_audio_dir, output_spec_dir, 
                  segment_duration, sr, save_segments, use_gpu):
    """
    Process all audio files in a genre directory with resume capability.
    """
    # Create genre-specific output directories
    if genre_name:
        audio_genre_dir = os.path.join(output_audio_dir, genre_name)
        spec_genre_dir = os.path.join(output_spec_dir, genre_name)
    else:
        audio_genre_dir = output_audio_dir
        spec_genre_dir = output_spec_dir
    
    os.makedirs(audio_genre_dir, exist_ok=True)
    os.makedirs(spec_genre_dir, exist_ok=True)
    
    # Get all audio files
    genre_path = Path(genre_path)
    audio_files = list(genre_path.glob('*.wav')) + list(genre_path.glob('*.au'))
    
    # Choose spectrogram function based on GPU availability
    spec_function = audio_to_spectrogram_gpu if (use_gpu and torch.cuda.is_available()) else "audio_to_spectrogram"
    
    total_segments = 0
    skipped_files = 0
    corrupted_files = []
    processed_files = 0
    
    for audio_file in audio_files:
        # Check if this file has already been fully processed
        base_name = audio_file.stem
        expected_segments = 10  # 30 seconds / 3 seconds = 10 segments
        
        # Check if all spectrograms for this file already exist
        all_exist = True
        for i in range(expected_segments):
            spec_filename = f"{base_name}_segment_{i:02d}.png"
            spec_path = os.path.join(spec_genre_dir, spec_filename)
            if not os.path.exists(spec_path):
                all_exist = False
                break
        
        if all_exist:
            print(f"  ✓ Skipping (already processed): {audio_file.name}")
            skipped_files += 1
            continue
        
        print(f"  Processing: {audio_file.name}")
        
        try:
            # Split audio into segments
            segment_paths = split_audio(str(audio_file), audio_genre_dir, 
                                        segment_duration, sr)
            
            if not segment_paths:  # Empty list returned (corrupted file)
                corrupted_files.append(audio_file.name)
                continue
            
            # Create spectrograms for each segment
            for segment_path in segment_paths:
                spec_filename = Path(segment_path).stem + '.png'
                spec_path = os.path.join(spec_genre_dir, spec_filename)
                
                # Skip if spectrogram already exists
                if os.path.exists(spec_path):
                    print(f"    ✓ Spectrogram exists: {spec_filename}")
                else:
                    spec_function(segment_path, spec_path, sr)
                
                # Delete segment audio file if not saving
                if not save_segments:
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
            
            total_segments += len(segment_paths)
            processed_files += 1
            
        except Exception as e:
            print(f"  ⚠️ ERROR processing {audio_file.name}: {str(e)}")
            corrupted_files.append(audio_file.name)
            continue
    
    # Print summary
    genre_label = f" ({genre_name})" if genre_name else ""
    print(f"\n  Summary{genre_label}:")
    print(f"    ✓ Processed: {processed_files} files ({total_segments} segments)")
    print(f"    ⏭️  Skipped (already done): {skipped_files} files")
    if corrupted_files:
        print(f"    ⚠️  Corrupted/Failed: {len(corrupted_files)} files")
        for corrupted in corrupted_files:
            print(f"       - {corrupted}")

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "genres_original"  # Change this to your GTZAN dataset path
    OUTPUT_AUDIO_DIR = "output_audio"
    OUTPUT_SPEC_DIR = "output_img"
    
    SEGMENT_DURATION = 3  # seconds
    SAMPLE_RATE = 22050
    SAVE_AUDIO_SEGMENTS = True  # Set to False if you only want spectrograms
    USE_GPU = True  # Set to True to use GPU acceleration for spectrograms
    
    print("GTZAN Audio Processing")
    print("=" * 50)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Segment duration: {SEGMENT_DURATION} seconds")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Save audio segments: {SAVE_AUDIO_SEGMENTS}")
    print(f"GPU acceleration: {USE_GPU and torch.cuda.is_available()}")
    print("=" * 50)
    
    # Process dataset
    process_gtzan_dataset(
        input_dir=INPUT_DIR,
        output_audio_dir=OUTPUT_AUDIO_DIR,
        output_spec_dir=OUTPUT_SPEC_DIR,
        segment_duration=SEGMENT_DURATION,
        sr=SAMPLE_RATE,
        save_segments=SAVE_AUDIO_SEGMENTS,
        use_gpu=USE_GPU
    )