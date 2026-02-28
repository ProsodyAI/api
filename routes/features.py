"""
Feature extraction endpoints.
"""

import logging
from typing import Optional
import base64
import os
import tempfile
import uuid

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field

from config import settings
from schemas import ProsodyFeaturesResponse, PhoneticFeaturesResponse

router = APIRouter()


# Lazy-load extractors
_prosody_extractor = None
_phonetic_extractor = None


def get_prosody_extractor():
    """Get or create prosody feature extractor."""
    global _prosody_extractor
    if _prosody_extractor is None:
        from prosody_ssm import ProsodyFeatureExtractor
        _prosody_extractor = ProsodyFeatureExtractor(sample_rate=settings.sample_rate)
    return _prosody_extractor


def get_phonetic_extractor():
    """Get or create phonetic feature extractor."""
    global _phonetic_extractor
    if _phonetic_extractor is None:
        from prosody_ssm import PhoneticFeatureExtractor
        _phonetic_extractor = PhoneticFeatureExtractor()
    return _phonetic_extractor


@router.post(
    "/prosody",
    response_model=ProsodyFeaturesResponse,
    summary="Extract prosody features",
    description="Extract prosodic features from audio without emotion classification.",
)
async def extract_prosody_features(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file"),
):
    """
    Extract prosodic features from audio.
    
    Features extracted:
    - Pitch (F0): mean, std, min, max, range
    - Energy: mean, std
    - Voice quality: jitter, shimmer, HNR
    - Rhythm: speech rate, pause rate
    - Spectral: centroid, rolloff, MFCCs
    """
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size // (1024*1024)}MB",
        )
    
    os.makedirs(settings.temp_dir, exist_ok=True)
    temp_path = os.path.join(settings.temp_dir, f"{uuid.uuid4()}{os.path.splitext(file.filename or '.wav')[1]}")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        extractor = get_prosody_extractor()
        features = extractor.extract_from_file(temp_path)
        
        return ProsodyFeaturesResponse(
            f0_mean=features.f0_mean,
            f0_std=features.f0_std,
            f0_min=features.f0_min,
            f0_max=features.f0_max,
            f0_range=features.f0_range,
            energy_mean=features.energy_mean,
            energy_std=features.energy_std,
            jitter=features.jitter,
            shimmer=features.shimmer,
            hnr=features.hnr,
            speech_rate=features.speech_rate,
            pause_rate=features.pause_rate,
            pause_duration_mean=features.pause_duration_mean,
            spectral_centroid_mean=features.spectral_centroid_mean,
            spectral_rolloff_mean=features.spectral_rolloff_mean,
            mfcc_means=features.mfcc_means.tolist() if hasattr(features.mfcc_means, 'tolist') else list(features.mfcc_means),
        )
    
    finally:
        background_tasks.add_task(cleanup_temp_file, temp_path)


class TextInput(BaseModel):
    """Input for phonetic feature extraction."""
    text: str = Field(..., description="Text to extract phonetic features from")
    language: str = Field(default="en-us", description="Language code")


@router.post(
    "/phonetic",
    response_model=PhoneticFeaturesResponse,
    summary="Extract phonetic features",
    description="Extract phonetic features from text.",
)
async def extract_phonetic_features(request: TextInput):
    """
    Extract phonetic features from text.
    
    Features extracted:
    - Phoneme list
    - Vowel/consonant ratios
    - Stressed syllable count
    """
    extractor = get_phonetic_extractor()
    
    try:
        features = extractor.extract_from_text(request.text)
        
        return PhoneticFeaturesResponse(
            phonemes=features.phonemes,
            vowel_ratio=features.vowel_ratio,
            consonant_ratio=features.consonant_ratio,
            stressed_syllable_count=features.stressed_syllable_count,
            phoneme_count=len(features.phonemes),
        )
    except Exception as e:
        logger.exception("Feature extraction failed")
        raise HTTPException(
            status_code=500,
            detail="Feature extraction failed. Please try again or contact support.",
        )


def cleanup_temp_file(path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
