"""
Qwen3-TTS Gradio App — MLX backend (Apple Silicon)

Setup:
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements_mlx.txt
    python mlx_app.py

Notes:
    - Requires an Apple Silicon (M-series) Mac; runs entirely on-device via
      `mlx-audio`, no CUDA/transformers involved.
    - Model weights are downloaded automatically on first use and cached
      locally afterwards.
"""

import tempfile

import gradio as gr
import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.tts.utils import load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# Display name (shown in the UI) -> speaker id expected by the model
CUSTOM_VOICE_SPEAKERS = {
    "Serena": "Serena",
    "Uncle Fu": "Uncle_Fu",
    "Vivian": "Vivian",
    "Aiden": "Aiden",
    "Ryan": "Ryan",
    "Ono Anna": "Ono_Anna",
    "Sohee": "Sohee",
    "Dylan": "Dylan",
    "Eric": "Eric",
}

# mlx-community repo ids, keyed by model size, for each model family
BASE_MODEL_IDS = {
    "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}
CUSTOM_VOICE_MODEL_IDS = {
    "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
    "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
}
# VoiceDesign is only released at 1.7B, so there is no size dropdown for it.
VOICE_DESIGN_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

# Fallback sample rate if a result object doesn't expose `.sample_rate`.
DEFAULT_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Model loading — cached so each model is loaded from disk/HF hub only once
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def get_model(model_id: str):
    """Load (or fetch from cache) an mlx-audio Qwen3-TTS model by repo id."""
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = load_model(model_id)
    return _MODEL_CACHE[model_id]


def _to_numpy(mx_audio) -> np.ndarray:
    """Convert an mlx `mx.array` audio buffer to numpy for soundfile.

    Going through `.tolist()` first (rather than a direct `np.array(mx_audio)`
    cast) is the conversion path confirmed to work reliably in practice.
    """
    if isinstance(mx_audio, mx.array):
        return np.array(mx_audio.tolist())
    return np.asarray(mx_audio)


def _save_wav(mx_audio, sr: int) -> str:
    """Write a generated waveform to a temp .wav file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, _to_numpy(mx_audio), sr)
    return tmp.name


def _friendly_error(e: Exception) -> str:
    """Turn an exception into a short, user-facing message (no traceback)."""
    msg = str(e)
    if "no such file" in msg.lower() or "not found" in msg.lower():
        return ("Model weights couldn't be found or downloaded. Check your internet "
                "connection, or that the model name is correct/accessible.")
    if "memory" in msg.lower():
        return "Ran out of memory. Try the 0.6B model, or shorten the text."
    return f"Generation failed: {msg[:300]}"


def _call_generate_with_optional_language(model, language, **kwargs):
    """Call the Base/clone `generate()` method, tolerating mlx-audio builds
    whose signature doesn't (yet) accept a `language` kwarg for this method.
    """
    try:
        return list(model.generate(language=language, **kwargs))
    except TypeError as e:
        if "language" in str(e):
            return list(model.generate(**kwargs))
        raise


# ---------------------------------------------------------------------------
# Tab-specific generation functions
# ---------------------------------------------------------------------------

def generate_clone(model_size, ref_audio, ref_text, text, language):
    """Clone tab: 3-second voice clone from an uploaded reference audio clip.

    A transcript of the reference audio (ref_text) is required — mlx-audio's
    Qwen3-TTS voice-clone model uses it to align the reference audio with
    its content. Leaving it out (or passing an empty string) is what causes
    stuttering / repeated words in the generated speech.
    """
    if not ref_audio:
        raise gr.Error("Please upload a reference audio clip first.")
    if not ref_text or not ref_text.strip():
        raise gr.Error(
            "Please enter a transcript of the reference audio. It's required for "
            "reliable cloning — without it, the output tends to stutter or "
            "repeat words."
        )
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the cloned voice to say.")

    try:
        model = get_model(BASE_MODEL_IDS[model_size])
        results = _call_generate_with_optional_language(
            model, language, text=text, ref_audio=ref_audio, ref_text=ref_text,
        )
        result = results[0]
        sr = getattr(result, "sample_rate", DEFAULT_SAMPLE_RATE)
        path = _save_wav(result.audio, sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


def generate_voice_design(description, text, language):
    """VoiceDesign tab: synthesize speech in a voice described in natural language."""
    if not description or not description.strip():
        raise gr.Error("Please describe the voice you want (age, tone, accent, personality...).")
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the designed voice to say.")

    try:
        model = get_model(VOICE_DESIGN_MODEL_ID)
        results = list(model.generate_voice_design(
            text=text,
            language=language,
            instruct=description,
        ))
        result = results[0]
        sr = getattr(result, "sample_rate", DEFAULT_SAMPLE_RATE)
        path = _save_wav(result.audio, sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


def generate_custom_voice(model_size, speaker_display, instruct, text, language):
    """CustomVoice tab: one of 9 built-in timbres, with 1.7B-only style control."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text for the voice to say.")

    speaker = CUSTOM_VOICE_SPEAKERS.get(speaker_display, speaker_display)
    # The 0.6B CustomVoice model doesn't support instruction control.
    instruct = instruct if model_size == "1.7B" else None

    try:
        model = get_model(CUSTOM_VOICE_MODEL_IDS[model_size])
        kwargs = dict(text=text, language=language, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct
        results = list(model.generate_custom_voice(**kwargs))
        result = results[0]
        sr = getattr(result, "sample_rate", DEFAULT_SAMPLE_RATE)
        path = _save_wav(result.audio, sr)
        return path, path
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_friendly_error(e))


# ---------------------------------------------------------------------------
# UI helpers — keep tab layout consistent and easy to extend with new tabs
# ---------------------------------------------------------------------------

def make_output_row():
    """Standard audio-output + download-button pair, shared by every tab."""
    audio_out = gr.Audio(label="Generated speech", type="filepath")
    download_btn = gr.DownloadButton(label="Download .wav", visible=True)
    return audio_out, download_btn


def toggle_instruct(model_size):
    """Show/enable the instruct box only for the 1.7B CustomVoice model."""
    is_large = model_size == "1.7B"
    return (
        gr.update(visible=is_large, interactive=is_large),
        gr.update(visible=not is_large),
    )


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-TTS (MLX / Apple Silicon)") as demo:
    gr.Markdown("# Qwen3-TTS — MLX backend (Apple Silicon)")
    gr.Markdown(
        "Voice cloning, voice design, and custom-voice generation, "
        "powered by `mlx-audio` on Apple Silicon."
    )

    with gr.Tabs():
        # -------------------------------------------------------------
        # Tab 1: Clone
        # -------------------------------------------------------------
        with gr.Tab("Clone"):
            with gr.Row():
                with gr.Column():
                    clone_size = gr.Dropdown(["0.6B", "1.7B"], value="0.6B", label="Model size")
                    clone_ref_audio = gr.Audio(label="Reference audio (~3 seconds)", type="filepath")
                    clone_ref_text = gr.Textbox(
                        label="Reference transcript (required)",
                        lines=2,
                        placeholder="Exact transcript of what's said in the reference audio clip.",
                    )
                    clone_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the cloned voice say?",
                    )
                    clone_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    gr.Markdown(
                        "_Tip: an accurate transcript is what makes cloning work well here — "
                        "a missing or wrong transcript is the most common cause of stuttering "
                        "or repeated words in the output._"
                    )
                    clone_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    clone_audio_out, clone_download = make_output_row()

            clone_btn.click(
                fn=generate_clone,
                inputs=[clone_size, clone_ref_audio, clone_ref_text, clone_text, clone_lang],
                outputs=[clone_audio_out, clone_download],
            )

        # -------------------------------------------------------------
        # Tab 2: VoiceDesign
        # -------------------------------------------------------------
        with gr.Tab("VoiceDesign"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Model: `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (only released at 1.7B).")
                    vd_description = gr.Textbox(
                        label="Voice description",
                        lines=4,
                        placeholder=(
                            "e.g. A warm, mellow older male voice, calm and reassuring, "
                            "slight British accent."
                        ),
                    )
                    vd_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the designed voice say?",
                    )
                    vd_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    vd_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    vd_audio_out, vd_download = make_output_row()

            vd_btn.click(
                fn=generate_voice_design,
                inputs=[vd_description, vd_text, vd_lang],
                outputs=[vd_audio_out, vd_download],
            )

        # -------------------------------------------------------------
        # Tab 3: CustomVoice
        # -------------------------------------------------------------
        with gr.Tab("CustomVoice"):
            with gr.Row():
                with gr.Column():
                    cv_size = gr.Dropdown(["0.6B", "1.7B"], value="1.7B", label="Model size")
                    cv_speaker = gr.Dropdown(
                        list(CUSTOM_VOICE_SPEAKERS.keys()), value="Serena", label="Timbre"
                    )
                    cv_instruct = gr.Textbox(
                        label="Style instruction (1.7B only)",
                        lines=2,
                        placeholder='e.g. "speak angrily", "whisper", "very slow"',
                        visible=True,
                        interactive=True,
                    )
                    cv_instruct_note = gr.Markdown(
                        "_The 0.6B CustomVoice model doesn't support style instructions "
                        "— switch to 1.7B to use this._",
                        visible=False,
                    )
                    cv_text = gr.Textbox(
                        label="Text to speak", lines=3,
                        placeholder="What should the voice say?",
                    )
                    cv_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    cv_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    cv_audio_out, cv_download = make_output_row()

            cv_size.change(
                fn=toggle_instruct,
                inputs=[cv_size],
                outputs=[cv_instruct, cv_instruct_note],
            )

            cv_btn.click(
                fn=generate_custom_voice,
                inputs=[cv_size, cv_speaker, cv_instruct, cv_text, cv_lang],
                outputs=[cv_audio_out, cv_download],
            )

        # -------------------------------------------------------------
        # Add future tabs here, following the same pattern as above:
        #   with gr.Tab("Long-form"): ...
        #   with gr.Tab("Transcribe"): ...
        #   with gr.Tab("Enhance"): ...
        # -------------------------------------------------------------

if __name__ == "__main__":
    demo.queue().launch()
