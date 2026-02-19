import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import librosa
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from collections import Counter
from pathlib import Path
import os

class MusicGenrePredictorAdvanced:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._print_gpu_info()
        with open(config_path, 'r') as f:
            results = json.load(f)
            self.class_names = results['class_names']
            self.model_type = results.get('model_type', 'resnet18')
        
        print(f"\n Model Configuration:")
        print(f"Classes: {self.class_names}")
        print(f"Model Type: {self.model_type}")
        print(f"\nLoading model...")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("=" * 70)
        print(" Predictor ready for inference!")
        print("=" * 70)
    
    def _print_gpu_info(self):
        print("\n" + "=" * 70)
        print("GPU INFORMATION")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total VRAM: {total_memory:.2f} GB")
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"Currently Allocated: {allocated:.4f} GB")
            print(f"Currently Reserved: {reserved:.4f} GB")
            
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            print("No GPU available - using CPU")
            print("Predictions will be slower on CPU")
        
        print("=" * 70)
    
    def _load_model(self, model_path):
        if self.model_type == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))
        elif self.model_type == "resnet34":
            model = models.resnet34(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Trained for {checkpoint['epoch']+1} epochs")
        return model
    
    def get_gpu_memory_usage(self):
        if not torch.cuda.is_available():
            return {
                'device': 'CPU',
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0
            }
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        return {
            'device': torch.cuda.get_device_name(0),
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': free
        }
    
    def print_gpu_usage(self, prefix=""):
        mem = self.get_gpu_memory_usage()
        if mem['device'] == 'CPU':
            print(f"{prefix}Running on CPU (no GPU memory used)")
        else:
            print(f"{prefix}GPU Memory Usage:")
            print(f"{prefix}  Allocated: {mem['allocated_gb']:.4f} GB / {mem['total_gb']:.2f} GB")
            print(f"{prefix}  Reserved:  {mem['reserved_gb']:.4f} GB")
            print(f"{prefix}  Free:      {mem['free_gb']:.2f} GB")
    
    def audio_to_spectrogram_from_array(self, y, sr):

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        temp_path = 'tmp/temp_spec.png'
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return Image.open(temp_path)

    def predict_from_spectrogram(self, spectrogram_image):
        mem_before = self.get_gpu_memory_usage()
        if spectrogram_image.mode != 'RGB':
            spectrogram_image = spectrogram_image.convert('RGB')
        img_tensor = self.transform(spectrogram_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        mem_after = self.get_gpu_memory_usage()
        
        predicted_genre = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        top_probs, top_indices = torch.topk(probabilities, min(3, len(self.class_names)))
        top_predictions = [
            (self.class_names[idx.item()], prob.item()) 
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
        
        return {
            'predicted_genre': predicted_genre,
            'confidence': confidence_score,
            'top_3_predictions': top_predictions,
            'all_probabilities': {
                self.class_names[i]: prob 
                for i, prob in enumerate(probabilities[0].cpu().numpy())
            },
            'gpu_memory_used': mem_after['allocated_gb'] - mem_before['allocated_gb']
        }
    
    def predict_single_segment(self, audio_path, offset=0, duration=3):
        y, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=duration)

        spec = self.audio_to_spectrogram_from_array(y, sr)
        
        return self.predict_from_spectrogram(spec)
    
    def predict_multi_segment(self, audio_path, num_segments=5, skip_intro_sec=10, skip_outro_sec=10):

        print("\n" + "=" * 70)
        print(" MULTI-SEGMENT GENRE PREDICTION")
        print("=" * 70)
        print(f"Audio File: {Path(audio_path).name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n Initial GPU State:")
        self.print_gpu_usage("   ")
        
        total_duration = librosa.get_duration(path=audio_path)
        print(f"\nTotal Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        start_skip = min(skip_intro_sec, total_duration * 0.1)
        end_skip = max(total_duration - skip_outro_sec, total_duration * 0.9)
        usable_duration = end_skip - start_skip
        
        if usable_duration < 3:
            print("   Song too short, using full duration")
            start_skip = 0
            usable_duration = total_duration
        
        print(f" Analyzing Range: {start_skip:.1f}s - {end_skip:.1f}s")
        print(f" Number of Segments: {num_segments}")
        print("\n" + "-" * 70)
        
        predictions = []
        confidences = []
        all_probs = []
        segment_times = []
        
        start_time = time.time()
        
        for i in range(num_segments):
            offset = start_skip + (i * usable_duration / num_segments)
            segment_times.append(offset)
            
            print(f"\nSegment {i+1}/{num_segments} ({offset:.1f}s - {offset+3:.1f}s):")
            
            segment_start = time.time()
            result = self.predict_single_segment(audio_path, offset=offset, duration=3)
            segment_time = time.time() - segment_start
            
            predictions.append(result['predicted_genre'])
            confidences.append(result['confidence'])
            all_probs.append(result['all_probabilities'])
            
            print(f"Prediction: {result['predicted_genre']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Time: {segment_time:.3f}s")
            if torch.cuda.is_available():
                print(f"GPU Memory Used: {result['gpu_memory_used']:.6f} GB")
        
        total_time = time.time() - start_time
        
        print("\n" + "-" * 70)
        print("  VOTING RESULTS")
        print("-" * 70)
        
        vote_counts = Counter(predictions)
        final_genre = vote_counts.most_common(1)[0][0]
        votes = vote_counts[final_genre]
        
        final_confidence = np.mean([
            c for p, c in zip(predictions, confidences) if p == final_genre
        ])
        
        avg_probs = {}
        for genre in self.class_names:
            avg_probs[genre] = np.mean([probs[genre] for probs in all_probs])
        
        sorted_genres = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n FINAL PREDICTION: {final_genre.upper()}")
        print(f"Confidence: {final_confidence*100:.2f}% (average of winning segments)")
        print(f"Votes: {votes}/{num_segments} ({votes/num_segments*100:.1f}%)")
        print(f"Total Time: {total_time:.2f}s ({total_time/num_segments:.2f}s per segment)")
        
        print(f"\n  Vote Distribution:")
        for genre, count in vote_counts.most_common():
            percentage = count / num_segments * 100
            bar = "█" * int(percentage / 5)  # Visual bar
            print(f"   {genre:12s}: {count}/{num_segments} ({percentage:5.1f}%) {bar}")
        
        print(f"\n Average Probabilities (across all segments):")
        for i, (genre, prob) in enumerate(sorted_genres[:5], 1):
            bar = "█" * int(prob * 50)  # Visual bar
            print(f"   {i}. {genre:12s}: {prob*100:5.2f}% {bar}")
        
        print(f"\nFinal GPU State:")
        self.print_gpu_usage("   ")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("=" * 70)
        
        return {
            'predicted_genre': final_genre,
            'confidence': final_confidence,
            'votes': votes,
            'total_segments': num_segments,
            'vote_percentage': votes / num_segments * 100,
            'all_predictions': predictions,
            'all_confidences': confidences,
            'vote_distribution': dict(vote_counts),
            'average_probabilities': avg_probs,
            'segment_times': segment_times,
            'total_time': total_time,
            'avg_time_per_segment': total_time / num_segments
        }
    
    def predict_from_audio(self, audio_path, mode='multi', num_segments=5):
        if mode == 'single':
            print(f"\n  Single-segment prediction from: {Path(audio_path).name}")
            result = self.predict_single_segment(audio_path)
            
            print(f"\n  Prediction: {result['predicted_genre'].upper()}")
            print(f"  Confidence: {result['confidence']*100:.2f}%")
            print(f"\n  Top 3:")
            for i, (genre, prob) in enumerate(result['top_3_predictions'], 1):
                print(f"    {i}. {genre}: {prob*100:.2f}%")
            
            return result
        else:
            return self.predict_multi_segment(audio_path, num_segments=num_segments)

def main():

    print("\n" + "=" * 70)
    print("MUSIC GENRE PREDICTOR - MULTI-SEGMENT WITH GPU TRACKING")
    print("=" * 70)
    
    predictor = MusicGenrePredictorAdvanced(
        model_path="../model/best_model_resnet18.pth",
        config_path="../Reports/results.json"
    )
    # import random
    # import os
    # from PIL import Image

    # # Test on ACTUAL training spectrograms
    # data_dir = "../dataset/output_img"

    # print("Testing model on GTZAN spectrograms:")
    # print("=" * 60)

    # correct = 0
    # total = 0
    # genres = ['blues', 'classical', 'jazz', 'metal', 'rock', 'disco', 'reggae']

    # for true_genre in genres:
    #     genre_path = os.path.join(data_dir, true_genre)
    #     if not os.path.exists(genre_path):
    #         continue
            
    #     specs = [f for f in os.listdir(genre_path) if f.endswith('.png')]
        
    #     # Test 3 random samples
    #     for i in range(min(3, len(specs))):
    #         spec_file = random.choice(specs)
    #         spec_path = os.path.join(genre_path, spec_file)
            
    #         img = Image.open(spec_path)
    #         result = predictor.predict_from_spectrogram(img)
            
    #         is_correct = result['predicted_genre'] == true_genre
    #         correct += is_correct
    #         total += 1
            
    #         symbol = "✓" if is_correct else "✗"
    #         print(f"{symbol} {true_genre:10s} → {result['predicted_genre']:10s} ({result['confidence']*100:.1f}%)")

    # print("=" * 60)
    # print(f"GTZAN Accuracy: {correct/total*100:.1f}%")

    # if correct/total < 0.7:
    #     print("\n❌ PROBLEM: Model doesn't even work on GTZAN!")
    #     print("   Issue is NOT modern music - it's the model or spectrograms")
    # else:
    #     print("\n✓ Model works on GTZAN")
    #     print("   Issue might be modern vs old music")
    
    print("EXAMPLE 1: Multi-Segment Prediction (Recommended)")
    
    audio_file = "../sample/Can You Feel My Heart.mp3" 
    
    if Path(audio_file).exists():
        result = predictor.predict_multi_segment(
            audio_file, 
            num_segments=5,  
            skip_intro_sec=10, 
            skip_outro_sec=10 
        )
        
        # Access results
        print(f"\n Result Summary:")
        print(f"   Genre: {result['predicted_genre']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"   Agreement: {result['vote_percentage']:.1f}%")
    else:
        print(f"\n   File not found: {audio_file}")
        print("Please update the path to your audio file")
    
    # Example 2: Single segment prediction (faster but less accurate)
    print("\n" + "🎯" * 35)
    print("EXAMPLE 2: Single-Segment Prediction (Faster)")
    print("🎯" * 35)
    
    if Path(audio_file).exists():
        result_single = predictor.predict_from_audio(audio_file, mode='single')
    
    # Example 3: Batch prediction on multiple files
    print("\n" + "🎯" * 35)
    print("EXAMPLE 3: Batch Prediction")
    print("🎯" * 35)
    
    audio_files = [
        "../sample/Can You Feel My Heart.mp3",
        "../sample/Fly Me To The Moon.mp3",
        "../sample/One Mississippi.mp3",
    ]
    
    # Filter existing files
    existing_files = [f for f in audio_files if Path(f).exists()]
    
    if existing_files:
        batch_results = []
        
        for audio_file in existing_files:
            result = predictor.predict_multi_segment(audio_file, num_segments=3)
            batch_results.append({
                'file': Path(audio_file).name,
                'genre': result['predicted_genre'],
                'confidence': result['confidence'],
                'votes': f"{result['votes']}/{result['total_segments']}"
            })
        
        print("\n" + "=" * 70)
        print("BATCH RESULTS SUMMARY")
        print("=" * 70)
        
        import pandas as pd
        df = pd.DataFrame(batch_results)
        print(df.to_string(index=False))
    else:
        print("\n   No audio files found for batch prediction")
        print("Update the audio_files list with your file paths")
    
    print("\n" + "=" * 70)
    print("  Prediction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()