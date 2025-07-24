<div align="center">
  
## 🎵 Higgs Audio v2 WebUI
[![Higgs Audio v2 Model](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)
[![Tokenizer](https://img.shields.io/badge/HuggingFace-Tokenizer-orange?logo=huggingface)](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)
[![Original Repo](https://img.shields.io/badge/GitHub-boson--ai%2Fhiggs--audio-black?logo=github)](https://github.com/boson-ai/higgs-audio)
[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-green?logo=huggingface)](https://huggingface.co/spaces/smola/higgs_audio_v2)

Generate high-quality speech from text with voice cloning, longform generation, multi-speaker generation, voice library, and smart batching—all in a user-friendly web interface.

<img width="2296" height="1260" alt="image" src="https://github.com/user-attachments/assets/5d12b595-c090-4943-bf33-89577e967e73" />
</div>

---

## 🚀 Features

- **One-click installer for Windows 10/11 (x64)**
  - Supports NVIDIA 30xx / 40xx / 50xx GPUs and CPU (slow)
  - **Recommended: 24GB VRAM** for best performance
- **Voice Cloning**: Upload your own voice or use predefined/library voices
- **Longform Generation**: Smart chunking for seamless long text synthesis
- **Multi-Speaker Generation**: Assign different voices to speakers, upload or select from library
- **Voice Library**: Save, manage, and reuse custom voices
- **Whisper Speech-to-Text**: Automatic captioning and voice sample transcription
- **Smart Voice Consistency**:
  - In longform generation, if Smart Voice is used, the first generated audio sample is used as a reference for all subsequent chunks to maintain speaker consistency.
  - In multi-speaker generation, the first generated audio for each speaker is used as a reference for that speaker's subsequent lines, ensuring consistent speaker identity.
- **Scene Description**: Optionally describe the environment for more expressive results
- **Comprehensive Debugging**: Installer and app provide clear, color-coded status and error messages

---

## 🖥️ Installation (Recommended: One-Click)

1. **[Download](https://github.com/Saganaki22/higgs-audio-WebUI/releases/tag/One-click-installer) and extract files in a folder then run** `run_installer.bat`
    - Walks you through all options: GPU/CPU selection, environment setup, dependencies
    - Handles everything: Conda, Python, PyTorch, requirements, Gradio, Whisper, etc.
    - If interrupted, just re-run—it will continue where it left off
2. **After installation, run** `run_gui.bat` to launch the web interface

> **Note:** The installer is for Windows 10/11 x64. For best results, use an NVIDIA 30xx/40xx/50xx GPU with at least 24GB VRAM. CPU mode is supported but much slower.

---

## ⚙️ Manual/Advanced Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Saganaki22/higgs-audio-WebUI.git
   cd higgs-audio-WebUI
   ```
2. **Create and activate Conda environment (Python 3.10):**
   ```bash
   conda create -n higgs_audio_env python=3.10
   conda activate higgs_audio_env
   ```
3. **Install requirements:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install gradio
   pip install faster-whisper
   ```
4. **Install PyTorch (choose one):**
   - **NVIDIA 30xx/40xx (CUDA 12.6):**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
     ```
   - **NVIDIA 50xx (CUDA 12.8):**
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```
   - **CPU (slow):**
     ```bash
     pip install torch torchvision torchaudio
     ```
5. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
6. **Run the app:**
   ```bash
   conda activate higgs_audio_env
   python higgs_audio_gradio.py
   ```

---

## 📝 Usage Highlights

- **Voice Cloning:** Upload a clear 10–30 second sample for best results. The app will auto-transcribe your sample using Whisper.
- **Longform Generation:** Paste long text; the app will smartly chunk it for natural synthesis. If Smart Voice is selected, the first chunk's audio is used as a reference for all subsequent chunks, ensuring consistent speaker identity.
- **Multi-Speaker Generation:** Use `[SPEAKER0]`, `[SPEAKER1]`, etc., tags in your transcript. You can upload distinct voices for each speaker or select from the voice library. The first generated audio for each speaker is used as a reference for their subsequent lines.
- **Voice Library:** Save and manage your favorite voices for quick reuse in any mode.
- **Scene Description:** Add context to your audio (e.g., "in a quiet room") for more expressive results.

---

## 📦 Model & Tokenizer

- **Model:** [Higgs Audio v2 Generation 3B Base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)
- **Tokenizer:** [Higgs Audio v2 Tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)
- **Original Model Repo:** [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)
- **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/smola/higgs_audio_v2)

---

## 📄 License

See [LICENSE](LICENSE) for details. Model and tokenizer are provided under their respective licenses by Boson AI.

---

## 🙏 Acknowledgements

- [Boson AI](https://github.com/boson-ai/higgs-audio) for the original model and tokenizer
- [HuggingFace](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) for model hosting
- [smola/higgs_audio_v2](https://huggingface.co/spaces/smola/higgs_audio_v2) for the live demo
