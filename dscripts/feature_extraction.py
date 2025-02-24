import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
import warnings
import soundfile as sf
from tqdm import tqdm
from dscripts.logger import DataLogger, get_logger

class FeatureExtractor:
    def __init__(self, sr: int = 22050, duration: float = 30.0, log_dir: Optional[Path] = None):
        self.sr = sr
        self.duration = duration
        self.log_dir = Path("logs") if log_dir is None else Path(log_dir)
        self.logger = get_logger("feature_extractor", self.log_dir)
        self.data_logger = DataLogger("feature_extraction", self.log_dir)
        
        # Track statistics
        self.processed_files = 0
        self.failed_files = 0
        self.feature_stats = {}
    
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extracts comprehensive audio features from an audio file.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Dictionary containing extracted features
        """
        self.logger.info(f"Processing file: {audio_path}")
        
        # Load audio
        try:
            y = self._load_audio(audio_path)
        except Exception as e:
            self.logger.error(f"Error loading audio: {str(e)}")
            raise
        
        features = {}
        try:
            # Spectral Features
            features.update(self._extract_spectral_features(y))
            
            # Rhythm Features
            features.update(self._extract_rhythm_features(y))
            
            # Mel Features
            features.update(self._extract_mel_features(y))
            
            # Harmonic Features
            features.update(self._extract_harmonic_features(y))
            
            # Energy Features
            features.update(self._extract_energy_features(y))
            
            # Validate features
            if not self.validate_features(features):
                self.logger.error(f"Feature validation failed for {audio_path}")
                raise ValueError("Feature validation failed")
            
            # Update feature statistics
            self._update_feature_stats(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
        
        return features
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            # Use soundfile as primary loader
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # Convert stereo to mono
            
            # Resample if necessary
            if sr != self.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            
            # Trim to duration if specified
            if self.duration:
                y = y[:int(self.duration * self.sr)]
                
        except Exception as e:
            self.logger.warning(f"Soundfile load failed, trying librosa: {str(e)}")
            try:
                # Try librosa as backup
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
            except Exception as e2:
                self.logger.error(f"Both loaders failed. Soundfile error: {str(e)}, Librosa error: {str(e2)}")
                raise RuntimeError("Failed to load audio file with both soundfile and librosa")
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        return y
    
    def _extract_spectral_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract enhanced spectral features."""
        # Original spectral features
        features = {
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=self.sr).mean(),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=self.sr).mean(),
            "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=self.sr).mean(),
            "spectral_contrast": np.mean(librosa.feature.spectral_contrast(y=y, sr=self.sr), axis=1).tolist(),
            "spectral_flatness": librosa.feature.spectral_flatness(y=y).mean()
        }
        
        # Additional spectral features
        features.update({
            "spectral_flux": np.mean(librosa.onset.onset_strength(y=y, sr=self.sr)),
            "spectral_spread": np.std(librosa.feature.spectral_centroid(y=y, sr=self.sr)),
            "spectral_skewness": self._compute_spectral_skewness(y),
            "spectral_kurtosis": self._compute_spectral_kurtosis(y),
            "spectral_slope": np.mean(librosa.feature.spectral_contrast(y=y, sr=self.sr, n_bands=1)),
            "spectral_decrease": self._compute_spectral_decrease(y)
        })
        
        return features
    
    def _compute_spectral_skewness(self, y: np.ndarray) -> float:
        """Compute spectral skewness."""
        spec = np.abs(librosa.stft(y))
        mean = np.mean(spec)
        std = np.std(spec)
        skewness = np.mean(((spec - mean) / (std + 1e-8)) ** 3)
        return float(skewness)
    
    def _compute_spectral_kurtosis(self, y: np.ndarray) -> float:
        """Compute spectral kurtosis."""
        spec = np.abs(librosa.stft(y))
        mean = np.mean(spec)
        std = np.std(spec)
        kurtosis = np.mean(((spec - mean) / (std + 1e-8)) ** 4) - 3
        return float(kurtosis)
    
    def _compute_spectral_decrease(self, y: np.ndarray) -> float:
        """Compute spectral decrease."""
        spec = np.abs(librosa.stft(y))
        freq_bins = np.arange(1, spec.shape[0])
        weights = 1.0 / freq_bins
        weighted_mean = np.sum(weights * np.diff(np.mean(spec, axis=1))) / np.sum(weights)
        return float(weighted_mean)
    
    def _extract_rhythm_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract enhanced rhythm features."""
        # Original rhythm features
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
        features = {
            "tempo": float(tempo),
            "beat_frames": beats.tolist(),
            "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y).mean()),
            "rhythm_strength": float(np.mean(librosa.onset.onset_strength(y=y, sr=self.sr)))
        }
        
        # Additional rhythm features
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=self.sr)
        
        features.update({
            "pulse_clarity": float(np.mean(pulse)),
            "tempo_entropy": float(self._compute_tempo_entropy(onset_env)),
            "rhythm_regularity": float(self._compute_rhythm_regularity(beats)),
            "beat_duration_stats": self._compute_beat_stats(beats, self.sr)
        })
        
        return features
    
    def _compute_tempo_entropy(self, onset_env: np.ndarray) -> float:
        """Compute entropy of tempo variations."""
        # Normalize onset envelope
        onset_env = onset_env - np.min(onset_env)
        onset_env = onset_env / (np.max(onset_env) + 1e-8)
        
        # Compute histogram
        hist, _ = np.histogram(onset_env, bins=50, density=True)
        hist = hist[hist > 0]
        
        # Compute entropy
        return -np.sum(hist * np.log2(hist + 1e-8))
    
    def _compute_rhythm_regularity(self, beats: np.ndarray) -> float:
        """Compute rhythm regularity based on beat intervals."""
        if len(beats) < 2:
            return 0.0
        
        intervals = np.diff(beats)
        regularity = 1.0 / (np.std(intervals) + 1e-8)
        return float(regularity)
    
    def _compute_beat_stats(self, beats: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute statistical measures of beat durations."""
        if len(beats) < 2:
            return {
                "mean_beat_duration": 0.0,
                "std_beat_duration": 0.0,
                "min_beat_duration": 0.0,
                "max_beat_duration": 0.0
            }
        
        intervals = np.diff(beats) / sr  # Convert to seconds
        return {
            "mean_beat_duration": float(np.mean(intervals)),
            "std_beat_duration": float(np.std(intervals)),
            "min_beat_duration": float(np.min(intervals)),
            "max_beat_duration": float(np.max(intervals))
        }
    
    def _extract_mel_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract mel-based features."""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        return {
            "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20), axis=1).tolist(),
            "mel_spec_mean": np.mean(mel_spec, axis=1).tolist(),
            "mel_spec_var": np.var(mel_spec, axis=1).tolist()
        }
    
    def _extract_harmonic_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract harmonic features."""
        harmonic, percussive = librosa.effects.hpss(y)
        return {
            "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=self.sr), axis=1).tolist(),
            "chroma_cens": np.mean(librosa.feature.chroma_cens(y=y, sr=self.sr), axis=1).tolist(),
            "tonnetz": np.mean(librosa.feature.tonnetz(y=harmonic, sr=self.sr), axis=1).tolist(),
            "harmonic_ratio": float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8))
        }
    
    def _extract_energy_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract energy-based features."""
        return {
            "rms": float(librosa.feature.rms(y=y).mean()),
            "energy": float(np.mean(y**2)),
            "energy_entropy": float(self._compute_entropy(y))
        }
    
    def _compute_entropy(self, y: np.ndarray) -> float:
        """Compute signal entropy."""
        hist, _ = np.histogram(y, bins=100, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _update_feature_stats(self, features: Dict[str, Any]):
        """Update running statistics for features."""
        for name, value in features.items():
            if isinstance(value, (int, float)):
                if name not in self.feature_stats:
                    self.feature_stats[name] = []
                self.feature_stats[name].append(value)
    
    def process_dataset(self, audio_dir: str, output_json: str, batch_size: int = 100) -> None:
        """Process entire dataset with progress tracking and error handling."""
        audio_files = list(Path(audio_dir).rglob("*.mp3"))
        feature_data = {}
        
        self.logger.info(f"Starting processing of {len(audio_files)} files")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        # Process files with progress bar
        for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
            try:
                features = self.extract_features(str(audio_file))
                feature_data[audio_file.stem] = features
                self.processed_files += 1
                
                # Save batch
                if i % batch_size == 0:
                    self._save_batch(feature_data, output_json)
                    self.logger.info(f"Saved batch {i//batch_size + 1}")
                
            except Exception as e:
                self.failed_files += 1
                self.logger.error(f"Failed to process {audio_file.stem}: {str(e)}")
                continue
        
        # Save final batch
        if feature_data:
            self._save_batch(feature_data, output_json)
        
        # Log final statistics
        self._log_final_stats()
    
    def _save_batch(self, feature_data: Dict[str, Any], output_json: str) -> None:
        """Save feature data with error handling."""
        try:
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    existing_data = json.load(f)
                feature_data.update(existing_data)
            
            with open(output_json, 'w') as f:
                json.dump(feature_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}")
            # Save backup
            backup_path = output_json + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(feature_data, f, indent=4)
            self.logger.info(f"Saved backup to {backup_path}")
    
    def _log_final_stats(self):
        """Log final processing statistics."""
        stats = {
            "total_files_processed": self.processed_files,
            "failed_files": self.failed_files,
            "success_rate": (self.processed_files - self.failed_files) / self.processed_files * 100
        }
        
        # Calculate feature statistics
        for name, values in self.feature_stats.items():
            if values:
                stats[f"{name}_mean"] = float(np.mean(values))
                stats[f"{name}_std"] = float(np.std(values))
        
        self.data_logger.metrics.update(stats)
        self.data_logger.save_metrics()
        
        self.logger.info("Feature extraction completed")
        self.logger.info(f"Processed {self.processed_files} files with {self.failed_files} failures")
        self.logger.info(f"Success rate: {stats['success_rate']:.2f}%")

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Validate extracted features."""
        # Check for NaN or infinite values
        for name, value in features.items():
            if isinstance(value, (float, int)):
                if np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Invalid value in feature {name}: {value}")
                    return False
            elif isinstance(value, list):
                if any(np.isnan(x) or np.isinf(x) for x in value):
                    self.logger.warning(f"Invalid values in feature array {name}")
                    return False
        
        # Check value ranges
        if features["tempo"] <= 0 or features["tempo"] > 300:
            self.logger.warning(f"Suspicious tempo value: {features['tempo']}")
            return False
        
        if features["spectral_centroid"] <= 0:
            self.logger.warning(f"Invalid spectral centroid: {features['spectral_centroid']}")
            return False
        
        return True

if __name__ == "__main__":
    AUDIO_DIR = "datasets/fma_small"
    OUTPUT_JSON = "data/audio_features.json"
    
    extractor = FeatureExtractor()
    extractor.process_dataset(AUDIO_DIR, OUTPUT_JSON)
