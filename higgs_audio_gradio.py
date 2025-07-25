import gradio as gr
import torch
import torchaudio
import os
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
import numpy as np
import tempfile
import re
import gc
import time
from datetime import datetime
from pydub import AudioSegment
from pydub.utils import which
import warnings

# Whisper for auto-transcription
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    print("✅ Using faster-whisper for transcription")
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("✅ Using openai-whisper for transcription")
    except ImportError:
        WHISPER_AVAILABLE = False
        print("⚠️ Whisper not available - voice samples will use dummy text")

# Initialize model
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global instances
serve_engine = None
whisper_model = None

# Cache management for optimizations
_audio_cache = {}
_token_cache = {}

def install_ffmpeg_if_needed():
    """Check if ffmpeg is available and provide installation instructions if not"""
    if which("ffmpeg") is None:
        print("⚠️ FFmpeg not found. For full audio format support, install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")
        return False
    return True

def convert_audio_to_standard_format(audio_path, target_sample_rate=24000, force_mono=False):
    """
    Convert any audio file to standard format using multiple fallback methods
    Returns: (audio_data_numpy, sample_rate) or raises exception
    Preserves stereo unless force_mono=True
    """
    print(f"🔄 Converting audio file: {audio_path}")
    
    # Method 1: Try torchaudio first (fastest)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono only if explicitly requested
        if force_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("🔄 Converted stereo to mono")
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Convert to numpy (preserve channel structure)
        if waveform.shape[0] == 1:
            # Mono - squeeze to 1D
            audio_data = waveform.squeeze().numpy()
        else:
            # Stereo - keep as 2D array (channels, samples)
            audio_data = waveform.numpy()
        
        channels = waveform.shape[0]
        samples = waveform.shape[1]
        print(f"✅ Loaded with torchaudio: {'stereo' if channels == 2 else 'mono'} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"⚠️ Torchaudio failed: {e}")
    
    # Method 2: Try pydub (handles more formats, especially MP3)
    try:
        # Load with pydub
        if audio_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(audio_path)
        else:
            # Try to auto-detect format
            audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono only if explicitly requested
        original_channels = audio.channels
        if force_mono and audio.channels > 1:
            audio = audio.set_channels(1)
            print("🔄 Converted stereo to mono")
        
        # Set sample rate
        audio = audio.set_frame_rate(target_sample_rate)
        
        # Convert to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1] range
        if audio.sample_width == 1:  # 8-bit
            audio_data = audio_data / 128.0
        elif audio.sample_width == 2:  # 16-bit
            audio_data = audio_data / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            audio_data = audio_data / 2147483648.0
        else:
            # Assume already normalized or unknown format
            audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 1 else audio_data
        
        # Handle stereo data (pydub gives interleaved samples)
        if audio.channels == 2 and not force_mono:
            # Reshape interleaved stereo data to (2, samples)
            audio_data = audio_data.reshape(-1, 2).T
        
        channel_info = f"{'stereo' if audio.channels == 2 and not force_mono else 'mono'}"
        print(f"✅ Loaded with pydub: {channel_info} - {len(audio_data)} samples at {target_sample_rate}Hz")
        return audio_data, target_sample_rate
        
    except Exception as e:
        print(f"⚠️ Pydub failed: {e}")
    
    # Method 3: Try scipy as final fallback
    try:
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono
        if len(audio_data.shape) > 1:
            if force_mono:
                # Convert stereo to mono
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            else:
                # Keep stereo, transpose to (channels, samples)
                audio_data = audio_data.T
        
        # Resample if needed (basic resampling)
        if sample_rate != target_sample_rate:
            # Simple resampling - for better quality, use librosa
            ratio = target_sample_rate / sample_rate
            if len(audio_data.shape) == 1:
                # Mono
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            else:
                # Stereo
                new_length = int(audio_data.shape[1] * ratio)
                resampled = np.zeros((audio_data.shape[0], new_length))
                for channel in range(audio_data.shape[0]):
                    resampled[channel] = np.interp(
                        np.linspace(0, audio_data.shape[1], new_length),
                        np.arange(audio_data.shape[1]),
                        audio_data[channel]
                    )
                audio_data = resampled
            sample_rate = target_sample_rate
        
        channel_info = f"{'stereo' if len(audio_data.shape) > 1 else 'mono'}"
        samples = audio_data.shape[1] if len(audio_data.shape) > 1 else len(audio_data)
        print(f"✅ Loaded with scipy: {channel_info} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"⚠️ Scipy failed: {e}")
    
    raise ValueError(f"❌ Could not load audio file: {audio_path}. Tried torchaudio, pydub, and scipy.")

def save_temp_audio_robust(audio_data, sample_rate, force_mono=False):
    """
    Robust version of save_temp_audio_fixed that handles various input formats
    and ensures compatibility with soundfile. Preserves stereo unless force_mono=True
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        elif not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure float32 dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono conversion
        if len(audio_data.shape) == 1:
            # Mono, shape (N,)
            if force_mono:
                audio_data = audio_data
            else:
                audio_data = np.expand_dims(audio_data, axis=0)  # (1, N)
        elif len(audio_data.shape) == 2:
            # Could be (channels, samples) or (samples, channels)
            if audio_data.shape[0] > audio_data.shape[1]:
                # (samples, channels) -> (channels, samples)
                audio_data = audio_data.T
            # If force_mono, average channels
            if force_mono and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0, keepdims=True)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Convert to tensor for torchaudio
        waveform = torch.from_numpy(audio_data).float()
        
        # Ensure 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        print(f"Saving audio: shape={waveform.shape}, dtype={waveform.dtype}, max={waveform.max()}, min={waveform.min()}")
        
        torchaudio.save(temp_path, waveform, sample_rate)
        
        print(f"✅ Saved audio to: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        # Cleanup temp file on error
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

def process_uploaded_audio(uploaded_audio, force_mono=False):
    """
    Process uploaded audio from Gradio, handling various formats
    Returns: (audio_data_numpy, sample_rate) ready for use
    Preserves stereo unless force_mono=True
    """
    if uploaded_audio is None:
        raise ValueError("No audio uploaded")
    
    sample_rate, audio_data = uploaded_audio
    
    # If audio_data is already numpy array from Gradio
    if isinstance(audio_data, np.ndarray):
        # Ensure float32
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono
        if len(audio_data.shape) > 1:
            if force_mono:
                # Convert stereo to mono
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            else:
                # Keep stereo, but ensure proper channel order (channels, samples)
                if audio_data.shape[1] < audio_data.shape[0]:
                    # Data is (samples, channels), transpose to (channels, samples)
                    audio_data = audio_data.T
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        channel_info = "mono"
        if len(audio_data.shape) > 1:
            channel_info = f"stereo ({audio_data.shape[0]} channels)"
        elif not force_mono and len(audio_data.shape) == 1:
            channel_info = "mono"
            
        samples = audio_data.shape[1] if len(audio_data.shape) > 1 else len(audio_data)
        print(f"✅ Processed uploaded audio: {channel_info} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
    
    else:
        raise ValueError("Unexpected audio data format from Gradio")

def enhanced_save_temp_audio_fixed(uploaded_voice, force_mono=False):
    """
    Enhanced version that replaces the original save_temp_audio_fixed function
    Preserves stereo unless force_mono=True
    """
    if uploaded_voice is None or len(uploaded_voice) != 2:
        raise ValueError("Invalid uploaded voice format")
    
    sample_rate, audio_data = uploaded_voice
    
    # Process the uploaded audio
    processed_audio, processed_rate = process_uploaded_audio(uploaded_voice, force_mono)
    
    # Save to temporary file
    return save_temp_audio_robust(processed_audio, processed_rate, force_mono)

def load_audio_file_robust(file_path, target_sample_rate=24000):
    """
    Load any audio file and convert to standard format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    return convert_audio_to_standard_format(file_path, target_sample_rate)

def check_dependencies():
    """Check and report on available audio processing libraries"""
    print("🔍 Checking audio processing dependencies...")
    
    dependencies = {
        "torchaudio": True,  # Should always be available in your setup
        "pydub": False,
        "scipy": False,
        "ffmpeg": False
    }
    
    try:
        import pydub
        dependencies["pydub"] = True
        print("✅ pydub available")
    except ImportError:
        print("⚠️ pydub not available - install with: pip install pydub")
    
    try:
        import scipy.io
        dependencies["scipy"] = True
        print("✅ scipy available")
    except ImportError:
        print("⚠️ scipy not available - install with: pip install scipy")
    
    dependencies["ffmpeg"] = install_ffmpeg_if_needed()
    
    return dependencies

def safe_audio_processing(uploaded_voice, operation_name):
    """Wrapper for safe audio processing with detailed error messages"""
    try:
        return enhanced_save_temp_audio_fixed(uploaded_voice)
    except Exception as e:
        error_msg = f"❌ Error processing audio for {operation_name}: {str(e)}\n"
        error_msg += "💡 Try these solutions:\n"
        error_msg += "  • Ensure your audio file is a valid WAV or MP3\n"
        error_msg += "  • Try converting your file using a different audio editor\n"
        error_msg += "  • Make sure the file isn't corrupted\n"
        error_msg += "  • Install additional dependencies: pip install pydub scipy"
        raise ValueError(error_msg)

def clear_caches():
    """Clear audio and token caches to free memory"""
    global _audio_cache, _token_cache
    _audio_cache.clear()
    _token_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleared caches and freed memory")

def get_cache_key(text, voice_ref=None, temperature=0.3):
    """Generate cache key for audio generation"""
    import hashlib
    key_str = f"{text}_{voice_ref}_{temperature}"
    return hashlib.md5(key_str.encode()).hexdigest()

# Create output directories - simplified
def create_output_directories():
    base_dirs = ["output/basic_generation", "output/voice_cloning", "output/longform_generation", "output/multi_speaker", "voice_library"]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)

# Initialize output directories
create_output_directories()

def get_output_path(category, filename_base, extension=".wav"):
    """Generate organized output paths with timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_base}{extension}"
    output_path = os.path.join("output", category, filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path

def save_transcript_if_enabled(transcript, category, filename_base):
    """Save transcript to file if whisper is available - DISABLED"""
    # Disabled - we don't want to permanently save transcripts
    return None

def save_audio_reference_if_enabled(audio_path, category, filename_base):
    """Save audio reference if whisper is available - DISABLED"""
    # Disabled - we don't want to permanently save audio references
    return None

# Voice library management
def get_voice_library_voices():
    """Get list of voices in the voice library"""
    voice_library_dir = "voice_library"
    voices = []
    if os.path.exists(voice_library_dir):
        for f in os.listdir(voice_library_dir):
            if f.endswith('.wav'):
                voice_name = f.replace('.wav', '')
                voices.append(voice_name)
    return voices

def save_voice_to_library(audio_data, sample_rate, voice_name):
    """Save a voice sample to the voice library"""
    if not voice_name or not voice_name.strip():
        return "❌ Please enter a voice name"
    
    voice_name = voice_name.strip().replace(' ', '_')
    voice_path = os.path.join("voice_library", f"{voice_name}.wav")
    
    # Check if voice already exists
    if os.path.exists(voice_path):
        return f"❌ Voice '{voice_name}' already exists in library"
    
    try:
        # Save audio using robust method
        temp_path = save_temp_audio_robust(audio_data, sample_rate)
        import shutil
        shutil.move(temp_path, voice_path)
        
        # Create transcript using Whisper
        transcription = transcribe_audio(voice_path)
        txt_path = voice_path.replace('.wav', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        return f"✅ Voice '{voice_name}' saved to library!"
    
    except Exception as e:
        return f"❌ Error saving voice: {str(e)}"

def delete_voice_from_library(voice_name):
    """Delete a voice from the library"""
    if not voice_name or voice_name == "None":
        return "❌ Please select a voice to delete"
    
    voice_path = os.path.join("voice_library", f"{voice_name}.wav")
    txt_path = os.path.join("voice_library", f"{voice_name}.txt")
    
    try:
        if os.path.exists(voice_path):
            os.remove(voice_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return f"✅ Voice '{voice_name}' deleted from library"
    except Exception as e:
        return f"❌ Error deleting voice: {str(e)}"

def get_all_available_voices():
    """Get combined list of predefined voices and voice library"""
    voice_prompts_dir = "examples/voice_prompts"
    predefined = [f for f in os.listdir(voice_prompts_dir) if f.endswith(('.wav', '.mp3'))] if os.path.exists(voice_prompts_dir) else []
    library = get_voice_library_voices()
    
    combined = ["None (Smart Voice)"]
    if predefined:
        combined.extend([f"📁 {voice}" for voice in predefined])
    if library:
        combined.extend([f"👤 {voice}" for voice in library])
    
    return combined

def get_voice_path(voice_selection):
    """Get the actual path for a voice selection"""
    if not voice_selection or voice_selection == "None (Smart Voice)":
        return None
    
    if voice_selection.startswith("📁 "):
        # Predefined voice
        voice_name = voice_selection[2:]
        return os.path.join(voice_prompts_dir, voice_name)
    elif voice_selection.startswith("👤 "):
        # Library voice
        voice_name = voice_selection[2:]
        return os.path.join("voice_library", f"{voice_name}.wav")
    
    return None

# Available voice prompts - this needs to be refreshed dynamically
voice_prompts_dir = "examples/voice_prompts"

def get_current_available_voices():
    """Get current available voices (refreshed each time)"""
    return get_all_available_voices()

available_voices = get_current_available_voices()

def initialize_model():
    global serve_engine
    if serve_engine is None:
        print("🚀 Initializing Higgs Audio model...")
        serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
        print("✅ Model initialized successfully")

def initialize_whisper():
    global whisper_model
    if whisper_model is None and WHISPER_AVAILABLE:
        try:
            # Try faster-whisper first
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Loaded faster-whisper model")
        except ImportError:
            # Fallback to openai-whisper
            import whisper
            whisper_model = whisper.load_model("base")
            print("✅ Loaded openai-whisper model")

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper"""
    if not WHISPER_AVAILABLE:
        return "This is a voice sample for cloning."
    
    try:
        initialize_whisper()
        
        if 'faster_whisper' in str(type(whisper_model)):
            # Using faster-whisper
            segments, info = whisper_model.transcribe(audio_path)
            transcription = " ".join([segment.text for segment in segments])
        else:
            # Using openai-whisper
            result = whisper_model.transcribe(audio_path)
            transcription = result["text"]
        
        # Clean up transcription
        transcription = transcription.strip()
        if not transcription:
            transcription = "This is a voice sample for cloning."
        
        print(f"🎤 Transcribed: {transcription[:100]}...")
        return transcription
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return "This is a voice sample for cloning."

def save_temp_audio_fixed(audio_data, sample_rate):
    """Enhanced version that handles any audio format"""
    return save_temp_audio_robust(audio_data, sample_rate)

def load_audio_file_robust(file_path, target_sample_rate=24000):
    """
    Load any audio file and convert to standard format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    return convert_audio_to_standard_format(file_path, target_sample_rate)

def create_voice_reference_txt(audio_path, transcript_sample=None):
    """Create a corresponding .txt file for the voice reference with auto-transcription"""
    txt_path = audio_path.replace('.wav', '.txt')
    
    if transcript_sample is None:
        # Auto-transcribe the audio
        transcript_sample = transcribe_audio(audio_path)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript_sample)
    
    print(f"📝 Created voice reference text: {txt_path}")
    return txt_path

def save_temp_audio(audio_data, sample_rate):
    """Save numpy audio data to temporary file and return path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    
    # Convert numpy array to tensor and save
    if isinstance(audio_data, np.ndarray):
        waveform = torch.from_numpy(audio_data).float()
        # If mono, add channel dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(temp_path, waveform, sample_rate)
    
    return temp_path

def parse_multi_speaker_text(text):
    """Parse multi-speaker text and extract speaker assignments"""
    # Look for [SPEAKER0], [SPEAKER1], etc.
    speaker_pattern = r'\[SPEAKER(\d+)\]\s*([^[]*?)(?=\[SPEAKER\d+\]|$)'
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    
    speakers = {}
    for speaker_id, content in matches:
        speaker_key = f"SPEAKER{speaker_id}"
        if speaker_key not in speakers:
            speakers[speaker_key] = []
        speakers[speaker_key].append(content.strip())
    
    return speakers

def auto_format_multi_speaker(text):
    """Auto-format text for multi-speaker if not already formatted"""
    # If already has speaker tags, return as-is
    if '[SPEAKER' in text:
        return text
    
    # Split by common dialogue indicators
    lines = text.split('\n')
    formatted_lines = []
    current_speaker = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for dialogue indicators
        if line.startswith('"') or line.startswith("'") or ':' in line:
            # Switch speakers for dialogue
            if len(formatted_lines) > 0:
                current_speaker = 1 - current_speaker
            formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
        else:
            # Regular text, assign to current speaker
            formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
    
    return '\n'.join(formatted_lines)

def smart_chunk_text(text, max_chunk_size=200):
    """Smart text chunking that respects sentence boundaries and paragraphs"""
    # First split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is short enough, keep it as one chunk
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue
        
        # Split long paragraphs by sentences
        sentences = []
        # Split by multiple sentence endings
        sentence_parts = re.split(r'([.!?]+)', paragraph)
        
        current_sentence = ""
        for i in range(0, len(sentence_parts), 2):
            if i < len(sentence_parts):
                current_sentence = sentence_parts[i].strip()
                if i + 1 < len(sentence_parts):
                    current_sentence += sentence_parts[i + 1]
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed limit, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=True):
    """Optimized audio generation with caching and incremental decoding"""
    cache_key = None
    if use_cache:
        cache_key = get_cache_key(str(messages), temperature=temperature)
        if cache_key in _audio_cache:
            print("🚀 Using cached audio result")
            return _audio_cache[cache_key]
    
    # Generate audio with optimizations
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    
    # Cache result if enabled
    if use_cache and cache_key:
        _audio_cache[cache_key] = output
        # Keep cache size manageable
        if len(_audio_cache) > 50:
            # Remove oldest entries
            oldest_key = next(iter(_audio_cache))
            del _audio_cache[oldest_key]
    
    return output

# VOICE LIBRARY FUNCTIONS

def test_voice_sample(audio_data, sample_rate, test_text="Hello, this is a test of my voice. How does it sound?"):
    """Test a voice sample with default text before saving to library"""
    if audio_data is None:
        return None, "❌ Please upload an audio sample first"
    
    try:
        # Initialize model
        initialize_model()
        
        # Save temporary audio using robust method
        temp_audio_path = save_temp_audio_robust(audio_data, sample_rate)
        temp_txt_path = create_voice_reference_txt(temp_audio_path)
        
        # Generate test audio using voice cloning
        system_content = "Generate audio following instruction."
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),
            Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
            Message(role="user", content=test_text)
        ]
        
        # Generate audio
        output = optimized_generate_audio(messages, 1024, 0.3, use_cache=False)
        
        # Save test output
        test_output_path = "voice_test_output.wav"
        torchaudio.save(test_output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        # Clean up temp files
        for path in [temp_audio_path, temp_txt_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        
        return test_output_path, "✅ Voice test completed! Listen to the result above."
        
    except Exception as e:
        return None, f"❌ Error testing voice: {str(e)}"

def generate_basic(
    transcript,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description
):
    # Initialize model if not already done
    initialize_model()
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Prepare system message
    system_content = "Generate audio following instruction."
    if scene_description and scene_description.strip():
        system_content += f"\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
    
    # Handle voice selection using the same method as voice cloning tab
    if voice_prompt and voice_prompt != "None (Smart Voice)":
        ref_audio_path = get_voice_path(voice_prompt)
        if ref_audio_path and os.path.exists(ref_audio_path):
            # Create dummy txt file if it doesn't exist
            txt_path = ref_audio_path.replace('.wav', '.txt').replace('.mp3', '.txt')
            if not os.path.exists(txt_path):
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("This is a voice sample.")
            
            # Use the same pattern as working voice cloning
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
                Message(role="user", content=transcript)
            ]
        else:
            # Fallback to smart voice if file doesn't exist
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=transcript)
            ]
    else:
        # Smart voice
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=transcript)
        ]
    
    # Generate audio with optimizations
    output = optimized_generate_audio(messages, max_new_tokens, temperature)
    
    # Save and return audio with organized output
    output_path = get_output_path("basic_generation", "basic_audio")
    torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    clear_caches()  # Clear cache after generation
    return output_path

def generate_voice_clone(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed
):
    # Initialize model if not already done
    initialize_model()
    
    # Validate inputs
    if not transcript.strip():
        raise ValueError("Please enter text to synthesize")
    
    if uploaded_voice is None or uploaded_voice[1] is None:
        raise ValueError("Please upload a voice sample for cloning")
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Save uploaded audio to temporary file using enhanced method
    temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
    temp_txt_path = None
    
    try:
        # Create corresponding txt file with auto-transcription
        temp_txt_path = create_voice_reference_txt(temp_audio_path)  # Auto-transcribes!
        
        # Use the same pattern as the official generation.py
        # The serve engine expects the voice reference format like this:
        system_content = "Generate audio following instruction."
        
        # Create messages similar to how the official code does it
        # First, add the voice reference as a user-assistant pair
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),  # Dummy prompt for voice ref
            Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
            Message(role="user", content=transcript)
        ]
        
        # Generate audio with optimizations
        output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=False)
        
        # Save and return audio with organized output
        output_path = get_output_path("voice_cloning", "cloned_voice")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        clear_caches()  # Clear cache after generation
        return output_path
    
    finally:
        # Clean up temporary files
        for path in [temp_audio_path, temp_txt_path]:
            if path and os.path.exists(path):
                os.unlink(path)

def generate_voice_clone_alternative(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed
):
    """Alternative voice cloning method using voice_ref format"""
    # Initialize model if not already done
    initialize_model()
    
    # Validate inputs
    if not transcript.strip():
        raise ValueError("Please enter text to synthesize")
    
    if uploaded_voice is None or uploaded_voice[1] is None:
        raise ValueError("Please upload a voice sample for cloning")
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Save uploaded audio to temporary file using enhanced method
    temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
    
    try:
        # Try the voice_ref format (this might be specific to newer versions)
        system_content = "Generate audio following instruction."
        
        # The format you were using - let's make sure the path is correct
        user_content = f"<|voice_ref_start|>\n{temp_audio_path}\n<|voice_ref_end|>\n\n{transcript}"
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]
        
        # Generate audio with optimizations
        output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=False)
        
        # Save and return audio with organized output
        output_path = get_output_path("voice_cloning", "cloned_voice_alt")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        clear_caches()  # Clear cache after generation
        return output_path
    
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

def generate_longform(
    transcript,
    voice_choice,
    uploaded_voice,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    chunk_size
):
    # Initialize model if not already done
    initialize_model()
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Smart chunking
    chunks = smart_chunk_text(transcript, max_chunk_size=chunk_size)
    
    # Handle voice reference setup
    temp_audio_path = None
    temp_txt_path = None
    voice_ref_path = None
    voice_ref_text = None
    first_chunk_audio_path = None
    first_chunk_text = None
    
    try:
        # Determine initial voice reference
        if voice_choice == "Upload Voice" and uploaded_voice is not None and uploaded_voice[1] is not None:
            temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
            temp_txt_path = create_voice_reference_txt(temp_audio_path)  # Auto-transcribes!
            voice_ref_path = temp_audio_path
            # Read transcription
            if temp_txt_path and os.path.exists(temp_txt_path):
                with open(temp_txt_path, 'r', encoding='utf-8') as f:
                    voice_ref_text = f.read().strip()
            else:
                voice_ref_text = "This is a voice sample for cloning."
        elif voice_choice == "Predefined Voice" and voice_prompt != "None (Smart Voice)":
            ref_audio_path = get_voice_path(voice_prompt)
            if ref_audio_path and os.path.exists(ref_audio_path):
                voice_ref_path = ref_audio_path
                # Ensure txt file exists for predefined voices
                txt_path = ref_audio_path.replace('.wav', '.txt').replace('.mp3', '.txt')
                if not os.path.exists(txt_path):
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write("This is a voice sample.")
                # Read transcription
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        voice_ref_text = f.read().strip()
                else:
                    voice_ref_text = "This is a voice sample."
        
        # Prepare system message
        system_content = "Generate audio following instruction."
        if scene_description and scene_description.strip():
            system_content += f"\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        # Generate audio for each chunk
        full_audio = []
        sampling_rate = 24000
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            if voice_choice == "Upload Voice" and voice_ref_path and voice_ref_text:
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)),
                    Message(role="user", content=chunk)
                ]
            elif voice_choice == "Predefined Voice" and voice_ref_path and voice_ref_text:
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)),
                    Message(role="user", content=chunk)
                ]
            elif voice_choice == "Smart Voice":
                if i == 0:
                    # First chunk: let model pick a voice
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=chunk)
                    ]
                else:
                    # Use first chunk's audio and text as reference for all subsequent chunks
                    if first_chunk_audio_path and first_chunk_text:
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_chunk_text),
                            Message(role="assistant", content=AudioContent(audio_url=first_chunk_audio_path)),
                            Message(role="user", content=chunk)
                        ]
                    else:
                        # Fallback if voice_ref_path or voice_ref_text is not available (shouldn't happen with Smart Voice)
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=chunk)
                        ]
            else:
                # Fallback for other voice choices
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=chunk)
                ]
            
            # Generate audio with optimizations
            output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=True)
            
            if voice_choice == "Smart Voice" and i == 0:
                # Save first chunk's audio and text for reference
                first_chunk_audio_path = f"first_chunk_audio_{seed}_{hash(transcript[:20])}.wav"
                torchaudio.save(first_chunk_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                first_chunk_text = chunk
            
            # Append audio
            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
        
        # Concatenate all audio chunks and save with organized output
        if full_audio:
            full_audio = np.concatenate(full_audio, axis=0)
            
            output_path = get_output_path("longform_generation", "longform_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio)[None, :], sampling_rate)
            clear_caches()  # Clear cache after generation
            return output_path
        else:
            clear_caches()
            return None
    
    finally:
        # Clean up temporary files
        cleanup_files = [temp_audio_path, temp_txt_path]
        for path in cleanup_files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass  # Ignore cleanup errors

# IMPROVED HANDLER FUNCTION FOR MULTI-SPEAKER
def handle_multi_speaker_generation(
    transcript, voice_method, speaker0_audio, speaker1_audio, speaker2_audio,
    speaker0_voice, speaker1_voice, speaker2_voice, temperature, max_new_tokens, 
    seed, scene_description, auto_format
):
    # Prepare uploaded voices list with better validation
    uploaded_voices = []
    if voice_method == "Upload Voices":
        for i, audio in enumerate([speaker0_audio, speaker1_audio, speaker2_audio]):
            if audio is not None and len(audio) == 2 and audio[1] is not None:
                # Validate audio data
                sample_rate, audio_data = audio
                if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                    uploaded_voices.append(audio)
                    print(f"✅ Added SPEAKER{i} audio: {len(audio_data)} samples at {sample_rate}Hz")
                else:
                    uploaded_voices.append(None)
                    print(f"⚠️ Invalid audio data for SPEAKER{i}")
            else:
                uploaded_voices.append(None)
                print(f"⚠️ No audio provided for SPEAKER{i}")
    
    # Prepare predefined voices list
    predefined_voices = []
    if voice_method == "Predefined Voices":
        predefined_voices = [speaker0_voice, speaker1_voice, speaker2_voice]
    
    return generate_multi_speaker(
        transcript, voice_method, uploaded_voices, predefined_voices,
        temperature, max_new_tokens, seed, scene_description, auto_format
    )

def generate_multi_speaker(
    transcript,
    voice_method,
    uploaded_voices,
    predefined_voices,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    auto_format
):
    # Initialize model if not already done
    initialize_model()
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Auto-format transcript if requested
    if auto_format:
        transcript = auto_format_multi_speaker(transcript)
    
    # Parse speaker assignments
    speakers = parse_multi_speaker_text(transcript)
    if not speakers:
        raise ValueError("No speakers found in transcript. Use [SPEAKER0], [SPEAKER1] format or enable auto-format.")
    
    print(f"🎭 Found speakers: {list(speakers.keys())}")
    
    # Prepare voice references
    voice_refs = {}
    temp_files = []
    speaker_audio_refs = {}  # For smart voice consistency
    # NEW: Store both first audio path and first text for each speaker (for Smart Voice)
    speaker_first_refs = {}  # {speaker_id: (audio_path, text_content)}
    # NEW: Store uploaded audio and transcription for each speaker (for Upload Voices)
    uploaded_voice_refs = {}  # {speaker_id: (audio_path, transcription)}
    
    try:
        if voice_method == "Upload Voices":
            # CRITICAL FIX: Handle uploaded voices properly for each speaker
            if uploaded_voices:
                for i, audio in enumerate(uploaded_voices):
                    if audio is not None and audio[1] is not None:
                        speaker_key = f"SPEAKER{i}"
                        print(f"🎤 Processing uploaded voice for {speaker_key}...")
                        print(f"📊 Audio data: {len(audio[1])} samples at {audio[0]}Hz")
                        # Save the uploaded audio properly using enhanced method
                        temp_path = enhanced_save_temp_audio_fixed(audio)
                        # CRITICAL: Create transcription for the voice reference
                        temp_txt_path = create_voice_reference_txt(temp_path)
                        # Read the transcription for use as reference text
                        if os.path.exists(temp_txt_path):
                            with open(temp_txt_path, 'r', encoding='utf-8') as f:
                                transcription = f.read().strip()
                        else:
                            transcription = "This is a voice sample for cloning."
                        # Store both audio path and transcription for this speaker
                        uploaded_voice_refs[speaker_key] = (temp_path, transcription)
                        temp_files.extend([temp_path, temp_txt_path])
                        print(f"✅ Setup voice reference for {speaker_key}: {temp_path}")
                        print(f"📝 Created transcription file: {temp_txt_path}")
                        print(f"📋 {speaker_key} transcription: '{transcription[:50]}...'")
            print(f"🎭 Upload Voices setup complete. Voice refs: {list(uploaded_voice_refs.keys())}")
                        
        elif voice_method == "Predefined Voices":
            # Handle predefined voices
            if predefined_voices:
                for i, voice_name in enumerate(predefined_voices):
                    if voice_name and voice_name != "None (Smart Voice)":
                        speaker_key = f"SPEAKER{i}"
                        ref_audio_path = get_voice_path(voice_name)
                        if ref_audio_path and os.path.exists(ref_audio_path):
                            voice_refs[speaker_key] = ref_audio_path
                            print(f"📁 Setup voice reference for {speaker_key}: {ref_audio_path}")
                            # Ensure txt file exists
                            txt_path = ref_audio_path.replace('.wav', '.txt').replace('.mp3', '.txt')
                            if not os.path.exists(txt_path):
                                with open(txt_path, 'w', encoding='utf-8') as f:
                                    f.write("This is a voice sample.")
        
        # Prepare system message - SAME AS WORKING VOICE CLONING
        system_content = "Generate audio following instruction."
        
        if scene_description and scene_description.strip():
            system_content += f"\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        # Generate audio for each speaker segment
        full_audio = []
        sampling_rate = 24000
        
        # Process transcript line by line
        lines = transcript.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line has speaker tag
            speaker_match = re.match(r'\[SPEAKER(\d+)\]\s*(.*)', line)
            if speaker_match:
                speaker_id = f"SPEAKER{speaker_match.group(1)}"
                text_content = speaker_match.group(2).strip()
                
                if not text_content:
                    continue
                
                print(f"🎭 Generating for {speaker_id}: {text_content[:50]}...")
                
                # CRITICAL FIX: Prepare messages based on voice method
                # This logic determines which voice reference to use for each speaker line
                
                if voice_method == "Upload Voices" and speaker_id in uploaded_voice_refs:
                    # UPLOAD VOICES: Always use the uploaded voice sample and its transcription as reference
                    ref_audio_path, ref_text = uploaded_voice_refs[speaker_id]
                    print(f"🎤 Using UPLOADED voice reference for {speaker_id}: {ref_audio_path} with text: '{ref_text}'")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=ref_text),
                        Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
                        Message(role="user", content=text_content)
                    ]
                    
                elif voice_method == "Predefined Voices" and speaker_id in voice_refs:
                    # PREDEFINED VOICES: Use the predefined voice sample as reference
                    print(f"📁 Using PREDEFINED voice reference for {speaker_id}: {voice_refs[speaker_id]}")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content="Please speak this text."),
                        Message(role="assistant", content=AudioContent(audio_url=voice_refs[speaker_id])),
                        Message(role="user", content=text_content)
                    ]
                    
                elif voice_method == "Smart Voice":
                    # SMART VOICE: Use consistency logic with auto-generated references
                    if speaker_id in speaker_first_refs:
                        # Use the FIRST generated audio and text for this speaker as reference
                        first_audio_path, first_text = speaker_first_refs[speaker_id]
                        print(f"🔄 Using FIRST SMART voice reference for {speaker_id}: {first_audio_path} with text: '{first_text}'")
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_text),
                            Message(role="assistant", content=AudioContent(audio_url=first_audio_path)),
                            Message(role="user", content=text_content)
                        ]
                    else:
                        # First time for this speaker - let model pick voice
                        print(f"🆕 First occurrence of {speaker_id} in SMART mode, letting AI pick voice")
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=text_content)
                        ]
                        
                else:
                    # FALLBACK: This should only happen if no voice reference is available
                    print(f"⚠️ FALLBACK: No voice reference found for {speaker_id} in {voice_method} mode")
                    print(f"📋 Available voice_refs: {list(voice_refs.keys())}")
                    print(f"📋 Available speaker_audio_refs: {list(speaker_audio_refs.keys())}")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=text_content)
                    ]
                
                print(f"📝 Generating audio for: '{text_content}'")
                
                # Generate audio with optimizations
                output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=False)
                
                # IMPORTANT: Smart Voice consistency logic ONLY applies to Smart Voice mode
                # For Upload Voices and Predefined Voices, we already have the voice references
                if voice_method == "Smart Voice" and speaker_id not in speaker_first_refs:
                    # Save the first generated audio and text for this speaker for future consistency
                    speaker_audio_path = f"temp_speaker_{speaker_id}_{seed}_{int(time.time())}.wav"
                    torchaudio.save(speaker_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                    # Small delay to ensure file is written
                    time.sleep(0.1)
                    # CRITICAL: Use auto-transcription for voice reference
                    transcribed_text = transcribe_audio(speaker_audio_path)
                    speaker_txt_path = speaker_audio_path.replace('.wav', '.txt')
                    with open(speaker_txt_path, 'w', encoding='utf-8') as f:
                        f.write(transcribed_text)
                    # Save both audio path and the first text_content
                    speaker_first_refs[speaker_id] = (speaker_audio_path, text_content)
                    temp_files.extend([speaker_audio_path, speaker_txt_path])
                    print(f"✅ Saved {speaker_id} FIRST SMART voice reference: {speaker_audio_path}")
                    print(f"📝 Auto-transcribed: '{transcribed_text[:50]}...'")
                    # Verify files exist
                    if os.path.exists(speaker_audio_path) and os.path.exists(speaker_txt_path):
                        print(f"✅ Voice reference files verified for {speaker_id}")
                    else:
                        print(f"⚠️ Warning: Voice reference files not created properly for {speaker_id}")
                
                # For Upload Voices and Predefined Voices, we DON'T save additional references
                # because we already have the uploaded/predefined voice samples
                
                # Validate output before adding
                if output.audio is not None and len(output.audio) > 0:
                    full_audio.append(output.audio)
                    sampling_rate = output.sampling_rate
                    print(f"✅ Added audio segment (length: {len(output.audio)} samples)")
                else:
                    print(f"⚠️ Empty or invalid audio output for: '{text_content}'")
                
                # Add a small pause between different speakers (not between same speaker)
                if len(full_audio) > 1:
                    # Check if this is a different speaker than the previous line
                    prev_line_idx = lines.index(line) - 1
                    if prev_line_idx >= 0:
                        prev_line = lines[prev_line_idx].strip()
                        if prev_line:
                            prev_match = re.match(r'\[SPEAKER(\d+)\]', prev_line)
                            if prev_match and prev_match.group(1) != speaker_match.group(1):
                                # Different speaker, add pause
                                pause_samples = int(0.3 * sampling_rate)  # 0.3 second pause
                                pause_audio = np.zeros(pause_samples, dtype=np.float32)
                                full_audio.append(pause_audio)
                                print(f"🔇 Added pause between speakers")
        
        # Concatenate all audio chunks and save with organized output
        if full_audio:
            full_audio = np.concatenate(full_audio, axis=0)
            
            output_path = get_output_path("multi_speaker", "multi_speaker_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio)[None, :], sampling_rate)
            
            print(f"🎉 Multi-speaker audio generated successfully: {output_path}")
            clear_caches()  # Clear cache after generation
            return output_path
        else:
            clear_caches()
            raise ValueError("No audio was generated. Check your transcript format and voice samples.")
    
    except Exception as e:
        print(f"❌ Error in multi-speaker generation: {e}")
        raise e
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"🧹 Cleaned up: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Could not clean up {temp_file}: {e}")

def refresh_voice_list():
    updated_voices = get_all_available_voices()
    return gr.update(choices=updated_voices)

def refresh_voice_list_multi():
    """Refresh voice list for multi-speaker (returns 3 updates)"""
    updated_voices = get_all_available_voices()
    return [gr.update(choices=updated_voices), gr.update(choices=updated_voices), gr.update(choices=updated_voices)]

def refresh_library_list():
    library_voices = ["None"] + get_voice_library_voices()
    return gr.update(choices=library_voices)

# Check audio processing capabilities at startup
check_dependencies()

# Gradio interface
with gr.Blocks(title="Higgs Audio v2 Generator") as demo:
    gr.HTML('<h1 style="text-align:center; margin-bottom:0.2em;"><a href="https://github.com/Saganaki22/higgs-audio-WebUI" target="_blank" style="text-decoration:none; color:inherit;">🎵 Higgs Audio v2 WebUI</a></h1>')
    gr.HTML('<div style="text-align:center; font-size:1.2em; margin-bottom:1.5em;">Generate high-quality speech from text with voice cloning, longform generation, multi speaker generation, voice library, smart batching</div>')
    with gr.Tabs():
        # Tab 1: Basic Generation with Predefined Voices
        with gr.Tab("Basic Generation"):
            with gr.Row():
                with gr.Column():
                    basic_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize...",
                        value="The sun rises in the east and sets in the west.",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Settings", open=True):
                        basic_voice_prompt = gr.Dropdown(
                            choices=available_voices,
                            value="None (Smart Voice)",
                            label="Predefined Voice Prompts"
                        )
                        basic_refresh_voices = gr.Button("Refresh Voice List")
                        
                        basic_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        basic_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        basic_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens"
                        )
                        basic_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                    
                    basic_generate_btn = gr.Button("Generate Audio", variant="primary")
                
                with gr.Column():
                    basic_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #ff9800;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Basic Generation:</b><br>
                        • For best results, use clear, natural sentences.<br>
                        • You can select a predefined voice or use Smart Voice for random high-quality voices.<br>
                        • Scene description can help set the environment (e.g., "in a quiet room").<br>
                        • Adjust temperature for more/less expressive speech.<br>
                        • Try different seeds for voice variety.
                    </div>
                    ''')
        
        # Tab 2: Voice Cloning (YOUR voice only)
        with gr.Tab("Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    vc_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize with your voice...",
                        value="Hello, this is my cloned voice speaking!",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Cloning", open=True):
                        gr.Markdown("### Upload Your Voice Sample")
                        vc_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                        if WHISPER_AVAILABLE:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Audio will be auto-transcribed!* ✨")
                        else:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        # Add a toggle to switch between methods
                        vc_method = gr.Radio(
                            choices=["Official Method", "Alternative Method"],
                            value="Official Method",
                            label="Voice Cloning Method"
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        vc_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        vc_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens"
                        )
                        vc_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                    
                    vc_generate_btn = gr.Button("Clone My Voice & Generate", variant="primary")
                
                with gr.Column():
                    vc_output_audio = gr.Audio(label="Cloned Voice Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #4caf50;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Voice Cloning:</b><br>
                        • Upload a clear 10-30 second sample of your voice, speaking naturally.<br>
                        • The sample will be auto-transcribed for best cloning results.<br>
                        • Use the "Official Method" for most cases; try "Alternative Method" if you want to experiment.<br>
                        • Longer, more expressive samples improve cloning quality.<br>
                        • Use the same seed to reproduce results.
                    </div>
                    ''')
        
        # Tab 3: Long-form Generation
        with gr.Tab("Long-form Generation"):
            with gr.Row():
                with gr.Column():
                    lf_transcript = gr.TextArea(
                        label="Long Transcript",
                        placeholder="Enter long text to synthesize...",
                        value="Artificial intelligence is transforming our world. It helps solve complex problems in healthcare, climate, and education. Machine learning algorithms can process vast amounts of data to find patterns humans might miss. As we develop these technologies, we must consider their ethical implications. The future of AI holds both incredible promise and significant challenges.",
                        lines=10
                    )
                    
                    with gr.Accordion("Voice Options", open=True):
                        lf_voice_choice = gr.Radio(
                            choices=["Smart Voice", "Upload Voice", "Predefined Voice"],
                            value="Smart Voice",
                            label="Voice Selection Method"
                        )
                        
                        with gr.Group(visible=False) as lf_upload_group:
                            lf_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                            if WHISPER_AVAILABLE:
                                gr.Markdown("*Audio will be auto-transcribed for voice cloning!* ✨")
                            else:
                                gr.Markdown("*Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        with gr.Group(visible=False) as lf_predefined_group:
                            lf_voice_prompt = gr.Dropdown(
                                choices=available_voices,
                                value="None (Smart Voice)",
                                label="Predefined Voice Prompts"
                            )
                            lf_refresh_voices = gr.Button("Refresh Voice List")
                        
                        lf_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        lf_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        lf_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens per Chunk"
                        )
                        lf_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                        lf_chunk_size = gr.Slider(
                            minimum=100,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Characters per Chunk"
                        )
                    
                    lf_generate_btn = gr.Button("Generate Long-form Audio", variant="primary")
                
                with gr.Column():
                    lf_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #2196f3;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Long-form Generation:</b><br>
                        • Paste or write long text (stories, articles, etc.) for continuous speech.<br>
                        • Choose Smart Voice, upload your own, or select a predefined voice.<br>
                        • Adjust chunk size for smoother transitions (smaller = more natural, larger = faster).<br>
                        • Scene description can set the mood or environment.<br>
                        • Use consistent voice references for best results in long texts.
                    </div>
                    ''')
            
            # Visibility logic for voice options
            def update_voice_options(choice):
                return {
                    lf_upload_group: gr.update(visible=choice == "Upload Voice"),
                    lf_predefined_group: gr.update(visible=choice == "Predefined Voice")
                }
            
            lf_voice_choice.change(
                fn=update_voice_options,
                inputs=lf_voice_choice,
                outputs=[lf_upload_group, lf_predefined_group]
            )
        
        # Tab 4: Multi-Speaker Generation
        with gr.Tab("Multi-Speaker Generation"):
            with gr.Row():
                with gr.Column():
                    ms_transcript = gr.TextArea(
                        label="Multi-Speaker Transcript",
                        placeholder="Enter text with [SPEAKER0], [SPEAKER1] tags or enable auto-format...",
                        value="[SPEAKER0] Hello there, how are you doing today?\n[SPEAKER1] I'm doing great, thank you for asking! How about yourself?\n[SPEAKER0] I'm fantastic! It's such a beautiful day outside.\n[SPEAKER1] Yes, it really is. Perfect weather for a walk in the park.",
                        lines=8
                    )
                    
                    with gr.Accordion("Voice Configuration", open=True):
                        ms_voice_method = gr.Radio(
                            choices=["Smart Voice", "Upload Voices", "Predefined Voices"],
                            value="Smart Voice",
                            label="Voice Method"
                        )
                        
                        ms_auto_format = gr.Checkbox(
                            label="Auto-format dialogue (converts regular text to speaker format)",
                            value=False
                        )
                        
                        with gr.Group(visible=False) as ms_upload_group:
                            gr.Markdown("### Upload Voice Samples")
                            gr.Markdown("*Upload distinct voice samples for each speaker. Each will be auto-transcribed for voice cloning.*")
                            ms_speaker0_audio = gr.Audio(label="SPEAKER0 Voice Sample", type="numpy")
                            ms_speaker1_audio = gr.Audio(label="SPEAKER1 Voice Sample", type="numpy")
                            ms_speaker2_audio = gr.Audio(label="SPEAKER2 Voice Sample (Optional)", type="numpy")
                            if WHISPER_AVAILABLE:
                                gr.Markdown("*✨ Voice samples will be auto-transcribed for perfect voice cloning!*")
                        
                        with gr.Group(visible=False) as ms_predefined_group:
                            gr.Markdown("### Select Predefined Voices")
                            ms_speaker0_voice = gr.Dropdown(
                                choices=get_current_available_voices(),
                                value="None (Smart Voice)",
                                label="SPEAKER0 Voice"
                            )
                            ms_speaker1_voice = gr.Dropdown(
                                choices=get_current_available_voices(),
                                value="None (Smart Voice)",
                                label="SPEAKER1 Voice"
                            )
                            ms_speaker2_voice = gr.Dropdown(
                                choices=get_current_available_voices(),
                                value="None (Smart Voice)",
                                label="SPEAKER2 Voice (Optional)"
                            )
                            ms_refresh_voices = gr.Button("Refresh Voice List")
                        
                        ms_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the conversation setting...",
                            value="A friendly conversation between two people in a quiet room."
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        ms_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        ms_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens per Segment"
                        )
                        ms_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                    
                    ms_generate_btn = gr.Button("Generate Multi-Speaker Audio", variant="primary")
                
                with gr.Column():
                    ms_output_audio = gr.Audio(label="Generated Multi-Speaker Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #e91e63;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Multi-Speaker Generation:</b><br>
                        • Use [SPEAKER0], [SPEAKER1], etc. tags to assign lines to speakers.<br>
                        • "Smart Voice" lets the AI pick distinct voices for each speaker.<br>
                        • "Upload Voices" allows you to clone multiple real voices (one per speaker).<br>
                        • "Predefined Voices" lets you pick from the voice library for each speaker.<br>
                        • Enable auto-format to convert plain dialogue into speaker format.<br>
                        • Scene description can set the conversation context.<br>
                        • For best results, upload clear, expressive samples for each speaker.
                    </div>
                    ''')
            
            # Visibility logic for voice method options
            def update_ms_voice_options(choice):
                return {
                    ms_upload_group: gr.update(visible=choice == "Upload Voices"),
                    ms_predefined_group: gr.update(visible=choice == "Predefined Voices")
                }
            
            ms_voice_method.change(
                fn=update_ms_voice_options,
                inputs=ms_voice_method,
                outputs=[ms_upload_group, ms_predefined_group]
            )
        
        # Tab 5: Voice Library Management
        with gr.Tab("Voice Library"):
            gr.HTML("<h2 style='text-align: center;'>🎵 Voice Library Management</h2>")
            gr.HTML("<p style='text-align: center;'>Save your voice samples to reuse across all generation modes!</p>")
            
            with gr.Row():
                with gr.Column():
                    # Step 1: Upload
                    gr.Markdown("### 🎤 Step 1: Upload Voice Sample")
                    vl_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="numpy")
                    
                    # Step 2: Test
                    gr.Markdown("### 🎵 Step 2: Test Voice Clone")
                    gr.Markdown("**Enter text below and test how your voice will sound!**")
                    
                    vl_test_text = gr.Textbox(
                        label="Test Text", 
                        placeholder="Enter text to test with your voice...",
                        value="This is a test of my voice cloning. How does it sound? I can use this voice for speech generation.",
                        lines=3
                    )
                    
                    with gr.Row():
                        vl_test_btn = gr.Button("🎵 TEST VOICE CLONE", variant="primary", size="lg")
                        vl_clear_test_btn = gr.Button("🔄 Clear Test", variant="secondary")
                    
                    # Step 3: Save
                    gr.Markdown("### 💾 Step 3: Save to Library")
                    vl_new_voice_name = gr.Textbox(
                        label="Voice Name", 
                        placeholder="Enter a name for this voice..."
                    )
                    vl_save_btn = gr.Button("💾 Save to Library", variant="stop", size="lg")
                    
                    if WHISPER_AVAILABLE:
                        gr.HTML("<p><em>✨ Voice will be auto-transcribed when saved!</em></p>")
                
                with gr.Column():
                    # Results Section
                    vl_test_audio = gr.Audio(label="🎧 Voice Test Result", type="filepath", show_download_button=True)
                    vl_test_status = gr.Textbox(label="Test Status", interactive=False)
                    vl_save_status = gr.Textbox(label="Save Status", interactive=False)
                    
                    # Library Display
                    def display_voice_library():
                        voices = get_voice_library_voices()
                        if not voices:
                            return "No voices in library yet. Add some voices to get started!"
                        
                        display_text = "**Your Voice Library:**\n\n"
                        for voice in voices:
                            voice_path = os.path.join("voice_library", f"{voice}.wav")
                            txt_path = os.path.join("voice_library", f"{voice}.txt")
                            
                            display_text += f"🎤 **{voice}**\n"
                            
                            if os.path.exists(txt_path):
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    transcript = f.read().strip()
                                    if len(transcript) > 100:
                                        transcript = transcript[:100] + "..."
                                    display_text += f"   📝 *{transcript}*\n"
                            
                            display_text += "\n"
                        
                        return display_text
                    
                    vl_library_display = gr.Markdown(display_voice_library())
            
            # Management Section
            gr.Markdown("---")
            gr.Markdown("### 🗂️ Manage Existing Voices")
            with gr.Row():
                with gr.Column():
                    vl_existing_voices = gr.Dropdown(
                        label="Select Voice to Delete",
                        choices=["None"] + get_voice_library_voices(),
                        value="None"
                    )
                with gr.Column():
                    with gr.Row():
                        vl_refresh_btn = gr.Button("Refresh List")
                        vl_delete_btn = gr.Button("Delete Selected", variant="stop")
            
            vl_delete_status = gr.Textbox(label="Delete Status", interactive=False)

    # Function to handle voice cloning method selection
    def handle_voice_clone_generation(
        transcript, uploaded_voice, temperature, max_new_tokens, seed, method
    ):
        if method == "Official Method":
            return generate_voice_clone(transcript, uploaded_voice, temperature, max_new_tokens, seed)
        else:
            return generate_voice_clone_alternative(transcript, uploaded_voice, temperature, max_new_tokens, seed)
    
    # Voice Library Event Handlers
    def handle_test_voice(audio_data, test_text):
        if audio_data is None or audio_data[1] is None:
            return None, "❌ Please upload an audio sample first"
        
        if not test_text.strip():
            test_text = "This is a test of my voice cloning."
        
        try:
            test_audio_path, status = test_voice_sample(audio_data[1], audio_data[0], test_text)
            return test_audio_path, status
        except Exception as e:
            return None, f"❌ Error testing voice: {str(e)}"
    
    def handle_clear_test():
        return None, "Test cleared. Upload voice and try again."
    
    def handle_save_voice(audio_data, voice_name):
        if audio_data is None or audio_data[1] is None:
            return "❌ Please upload an audio sample first", gr.update(), gr.update()
        
        if not voice_name or not voice_name.strip():
            return "❌ Please enter a voice name", gr.update(), gr.update()
        
        try:
            status = save_voice_to_library(audio_data[1], audio_data[0], voice_name)
            new_choices = ["None"] + get_voice_library_voices()
            new_display = display_voice_library()
            
            return status, gr.update(choices=new_choices), gr.update(value=new_display)
        except Exception as e:
            return f"❌ Error saving voice: {str(e)}", gr.update(), gr.update()
    
    def handle_delete_voice(voice_name):
        status = delete_voice_from_library(voice_name)
        new_choices = ["None"] + get_voice_library_voices()
        new_display = display_voice_library()
        
        return status, gr.update(choices=new_choices, value="None"), gr.update(value=new_display)
    
    def handle_refresh_library():
        new_choices = ["None"] + get_voice_library_voices()
        new_display = display_voice_library()
        return gr.update(choices=new_choices), gr.update(value=new_display)

    # Event handling for Basic Generation
    basic_generate_btn.click(
        fn=generate_basic,
        inputs=[basic_transcript, basic_voice_prompt, basic_temperature, basic_max_new_tokens, basic_seed, basic_scene_description],
        outputs=basic_output_audio
    )
    
    basic_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=basic_voice_prompt
    )
    
    # Event handling for Voice Cloning with method selection
    vc_generate_btn.click(
        fn=handle_voice_clone_generation,
        inputs=[vc_transcript, vc_uploaded_voice, vc_temperature, vc_max_new_tokens, vc_seed, vc_method],
        outputs=vc_output_audio
    )
    
    # Event handling for Long-form Generation
    lf_generate_btn.click(
        fn=generate_longform,
        inputs=[lf_transcript, lf_voice_choice, lf_uploaded_voice, lf_voice_prompt, lf_temperature, lf_max_new_tokens, lf_seed, lf_scene_description, lf_chunk_size],
        outputs=lf_output_audio
    )
    
    lf_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=lf_voice_prompt
    )
    
    # Event handling for Multi-Speaker Generation
    ms_generate_btn.click(
        fn=handle_multi_speaker_generation,
        inputs=[
            ms_transcript, ms_voice_method, ms_speaker0_audio, ms_speaker1_audio, ms_speaker2_audio,
            ms_speaker0_voice, ms_speaker1_voice, ms_speaker2_voice, ms_temperature, 
            ms_max_new_tokens, ms_seed, ms_scene_description, ms_auto_format
        ],
        outputs=ms_output_audio
    )
    
    ms_refresh_voices.click(
        fn=refresh_voice_list_multi,
        outputs=[ms_speaker0_voice, ms_speaker1_voice, ms_speaker2_voice]
    )
    
    # Event handling for Voice Library
    vl_test_btn.click(
        fn=handle_test_voice,
        inputs=[vl_new_voice_audio, vl_test_text],
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_clear_test_btn.click(
        fn=handle_clear_test,
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_save_btn.click(
        fn=handle_save_voice,
        inputs=[vl_new_voice_audio, vl_new_voice_name],
        outputs=[vl_save_status, vl_existing_voices, vl_library_display]
    )
    
    vl_delete_btn.click(
        fn=handle_delete_voice,
        inputs=[vl_existing_voices],
        outputs=[vl_delete_status, vl_existing_voices, vl_library_display]
    )
    
    vl_refresh_btn.click(
        fn=handle_refresh_library,
        outputs=[vl_existing_voices, vl_library_display]
    )

    # --- Place the GitHub link at the bottom of the app ---
    gr.HTML("""
    <div style='width:100%;text-align:center;margin-top:2em;margin-bottom:1em;'>
        <a href='https://github.com/Saganaki22/higgs-audio-WebUI' target='_blank' style='color:#fff;font-size:1.1em;text-decoration:underline;'>Github</a>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Higgs Audio v2 Generator...")
    print("✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing")
    demo.launch()
