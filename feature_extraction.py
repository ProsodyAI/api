"""
Prosodic and phonetic feature extraction (moved from prosody_ssm into API).

Extracts:
- F0 (fundamental frequency / pitch)
- Energy / intensity contours
- Voice quality (jitter, shimmer, HNR)
- Rhythm features (speech rate, pause patterns)
- Phonetic/phoneme features
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import librosa
    import scipy.signal as signal
except ImportError as e:
    raise ImportError(
        "Audio processing dependencies not installed. "
        "Run: pip install librosa scipy soundfile"
    ) from e


@dataclass
class ProsodyFeatures:
    """Container for extracted prosodic features."""

    # Pitch features
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_range: float = 0.0
    f0_contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # Energy features
    energy_mean: float = 0.0
    energy_std: float = 0.0
    energy_contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # Voice quality
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0  # Harmonics-to-Noise Ratio

    # Rhythm features
    speech_rate: float = 0.0  # syllables per second
    pause_rate: float = 0.0   # pauses per second
    pause_duration_mean: float = 0.0

    # Spectral features
    spectral_centroid_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    mfcc_means: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_vector(self) -> np.ndarray:
        """Convert features to a flat numpy array for model input."""
        scalar_features = np.array([
            self.f0_mean, self.f0_std, self.f0_min, self.f0_max, self.f0_range,
            self.energy_mean, self.energy_std,
            self.jitter, self.shimmer, self.hnr,
            self.speech_rate, self.pause_rate, self.pause_duration_mean,
            self.spectral_centroid_mean, self.spectral_rolloff_mean,
        ])
        return np.concatenate([scalar_features, self.mfcc_means])


@dataclass
class PhoneticFeatures:
    """Container for phonetic/phoneme-level features."""

    phonemes: list[str] = field(default_factory=list)
    phoneme_durations: list[float] = field(default_factory=list)
    vowel_ratio: float = 0.0
    consonant_ratio: float = 0.0
    stressed_syllable_count: int = 0
    phoneme_embeddings: Optional[np.ndarray] = None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        base = np.array([
            self.vowel_ratio,
            self.consonant_ratio,
            float(self.stressed_syllable_count),
            len(self.phonemes),
        ])
        if self.phoneme_embeddings is not None:
            return np.concatenate([base, self.phoneme_embeddings.mean(axis=0)])
        return base


class ProsodyFeatureExtractor:
    """
    Extract prosodic features from audio for emotion recognition.

    Features extracted:
    - Pitch (F0) statistics and contour
    - Energy/intensity patterns
    - Voice quality metrics (jitter, shimmer, HNR)
    - Rhythm features
    - Spectral features (MFCCs, centroid, rolloff)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(self, audio: np.ndarray, sr: Optional[int] = None) -> ProsodyFeatures:
        """
        Extract prosodic features from audio.

        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (uses self.sample_rate if None)

        Returns:
            ProsodyFeatures containing all extracted features
        """
        if sr is None:
            sr = self.sample_rate

        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        features = ProsodyFeatures()

        # Extract pitch (F0) using pyin
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        # Filter out unvoiced frames
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features.f0_mean = float(np.nanmean(f0_voiced))
            features.f0_std = float(np.nanstd(f0_voiced))
            features.f0_min = float(np.nanmin(f0_voiced))
            features.f0_max = float(np.nanmax(f0_voiced))
            features.f0_range = features.f0_max - features.f0_min
            features.f0_contour = f0

        # Extract energy (RMS)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        features.energy_mean = float(np.mean(rms))
        features.energy_std = float(np.std(rms))
        features.energy_contour = rms

        # Voice quality features
        features.jitter = self._compute_jitter(f0_voiced)
        features.shimmer = self._compute_shimmer(audio, sr)
        features.hnr = self._compute_hnr(audio, sr)

        # Rhythm features
        features.speech_rate, features.pause_rate, features.pause_duration_mean = \
            self._compute_rhythm_features(audio, sr, rms)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        features.spectral_centroid_mean = float(np.mean(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        features.spectral_rolloff_mean = float(np.mean(spectral_rolloff))

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        features.mfcc_means = np.mean(mfccs, axis=1)

        return features

    def _compute_jitter(self, f0: np.ndarray) -> float:
        """Compute jitter (pitch perturbation)."""
        if len(f0) < 2:
            return 0.0

        # Convert F0 to periods
        periods = 1.0 / f0[f0 > 0]
        if len(periods) < 2:
            return 0.0

        # Relative average perturbation
        period_diffs = np.abs(np.diff(periods))
        jitter = np.mean(period_diffs) / np.mean(periods)
        return float(jitter)

    def _compute_shimmer(self, audio: np.ndarray, sr: int) -> float:
        """Compute shimmer (amplitude perturbation)."""
        # Get amplitude envelope
        analytic = signal.hilbert(audio)
        envelope = np.abs(analytic)

        # Downsample envelope to frame rate
        frame_length_samples = int(0.025 * sr)  # 25ms frames
        hop_samples = int(0.010 * sr)  # 10ms hop

        n_frames = (len(envelope) - frame_length_samples) // hop_samples + 1
        if n_frames < 2:
            return 0.0

        amplitudes = np.array([
            np.max(envelope[i * hop_samples:i * hop_samples + frame_length_samples])
            for i in range(n_frames)
        ])

        # Relative average perturbation
        amp_diffs = np.abs(np.diff(amplitudes))
        shimmer = np.mean(amp_diffs) / (np.mean(amplitudes) + 1e-8)
        return float(shimmer)

    def _compute_hnr(self, audio: np.ndarray, sr: int) -> float:
        """Compute Harmonics-to-Noise Ratio."""
        # Use autocorrelation method
        frame_length = int(0.040 * sr)  # 40ms frame

        if len(audio) < frame_length:
            return 0.0

        # Take a frame from the middle of the audio
        start = len(audio) // 2 - frame_length // 2
        frame = audio[start:start + frame_length]

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Find the first peak after the origin (fundamental period)
        min_lag = int(sr / self.f0_max)
        max_lag = int(sr / self.f0_min)

        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1

        peak_region = autocorr[min_lag:max_lag]
        if len(peak_region) == 0:
            return 0.0

        peak_idx = np.argmax(peak_region) + min_lag
        peak_val = autocorr[peak_idx]
        origin_val = autocorr[0]

        if peak_val <= 0 or origin_val <= 0:
            return 0.0

        # HNR in dB
        hnr = 10 * np.log10(peak_val / (origin_val - peak_val + 1e-8) + 1e-8)
        return float(np.clip(hnr, -20, 40))

    def _compute_rhythm_features(
        self,
        audio: np.ndarray,
        sr: int,
        rms: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute rhythm-related features."""
        duration = len(audio) / sr

        # Detect speech/silence using energy threshold
        threshold = np.mean(rms) * 0.3
        is_speech = rms > threshold

        # Find pause regions
        pauses = []
        pause_start = None
        frame_duration = self.hop_length / sr

        for i, speech in enumerate(is_speech):
            if not speech and pause_start is None:
                pause_start = i
            elif speech and pause_start is not None:
                pause_duration = (i - pause_start) * frame_duration
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pauses.append(pause_duration)
                pause_start = None

        pause_rate = len(pauses) / duration if duration > 0 else 0.0
        pause_duration_mean = np.mean(pauses) if pauses else 0.0

        # Estimate speech rate using onset detection
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=self.hop_length)
        speech_rate = len(onsets) / duration if duration > 0 else 0.0

        return float(speech_rate), float(pause_rate), float(pause_duration_mean)

    def extract_from_file(self, audio_path: str) -> ProsodyFeatures:
        """Extract features from an audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self.extract(audio, sr)


class PhoneticFeatureExtractor:
    """
    Extract phonetic/phoneme-level features from text or audio.

    Uses phonemizer for text-to-phoneme conversion when available;
    otherwise falls back to a simple approximation.
    """

    # IPA vowels and consonants for feature computation
    VOWELS = set("aeiouɑɐɒæɛɜəɪɔøœʊʌɨʉɘɵɤʏ")
    CONSONANTS = set("bcdfghjklmnpqrstvwxyzʃʒθðŋɹɾʔβçʎɲɖɟɡɠɦʀʁχʂʐɕʑɻɭɬɮ")

    def __init__(self, language: str = "en-us", backend: str = "espeak"):
        self.language = language
        self.backend = backend
        self._phonemizer = None

    def _get_phonemizer(self):
        """Lazy-load phonemizer to avoid import issues if not installed."""
        if self._phonemizer is None:
            try:
                from phonemizer import phonemize
                from phonemizer.separator import Separator
                self._phonemize_fn = phonemize
                self._separator = Separator(phone=' ', word='|', syllable='-')
            except ImportError:
                self._phonemize_fn = None
                self._separator = None
        return self._phonemize_fn

    def extract_from_text(self, text: str) -> PhoneticFeatures:
        """Extract phonetic features from text."""
        features = PhoneticFeatures()

        phonemize_fn = self._get_phonemizer()

        if phonemize_fn is not None:
            try:
                phoneme_str = phonemize_fn(
                    text,
                    language=self.language,
                    backend=self.backend,
                    separator=self._separator,
                    strip=True,
                )
                features.phonemes = phoneme_str.split()
            except Exception:
                features.phonemes = self._simple_phonemize(text)
        else:
            features.phonemes = self._simple_phonemize(text)

        # Compute phoneme statistics
        if features.phonemes:
            total = len(features.phonemes)
            vowel_count = sum(1 for p in features.phonemes if any(c in self.VOWELS for c in p))
            consonant_count = sum(1 for p in features.phonemes if any(c in self.CONSONANTS for c in p))

            features.vowel_ratio = vowel_count / total
            features.consonant_ratio = consonant_count / total

            features.stressed_syllable_count = sum(
                1 for p in features.phonemes if 'ˈ' in p or "'" in p
            )

        return features

    def _simple_phonemize(self, text: str) -> list[str]:
        """Simple fallback phonemization without external dependencies."""
        phonemes = []
        text = text.lower()

        i = 0
        while i < len(text):
            char = text[i]

            if i < len(text) - 1:
                digraph = text[i:i+2]
                if digraph in ('th', 'sh', 'ch', 'ng', 'wh'):
                    phonemes.append(digraph)
                    i += 2
                    continue

            if char.isalpha():
                phonemes.append(char)
            elif char == ' ':
                phonemes.append('|')

            i += 1

        return phonemes

    def extract_with_alignments(
        self,
        text: str,
        word_alignments: list[tuple[str, float, float]],
    ) -> PhoneticFeatures:
        """Extract phonetic features with word-level timing from ASR."""
        features = self.extract_from_text(text)

        if word_alignments and features.phonemes:
            total_duration = sum(end - start for _, start, end in word_alignments)
            avg_phoneme_duration = total_duration / len(features.phonemes) if features.phonemes else 0
            features.phoneme_durations = [avg_phoneme_duration] * len(features.phonemes)

        return features
