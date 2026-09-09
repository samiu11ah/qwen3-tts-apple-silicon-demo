# 🎙️ Qwen3-TTS — Apple Silicon Demo

A simple, local voice-cloning, voice-design, and custom-voice generator powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), running entirely on your Mac using [MLX](https://github.com/ml-explore/mlx). No coding experience needed — just follow the steps below.

▶️ **Watch the full setup walkthrough on YouTube:** [Insert your video link here]

<!-- Add a screenshot or GIF of the app here once you have one, e.g.: -->
<!-- ![App screenshot](screenshot.png) -->

> 🖥️ On Windows or Linux with an NVIDIA GPU? Use the companion repo instead: [qwen3-tts-nvidia-demo](#https://github.com/samiu11ah/qwen3-tts-nvidia-demo.git)

---

## What can this do?

This app gives you a simple web interface (opens in your browser) with three tools:

- **Clone** — clone any voice from a short reference audio clip (plus its transcript) and make it say anything.
- **VoiceDesign** — describe a voice in plain English ("a warm, older male voice with a British accent") and the model builds it from scratch.
- **CustomVoice** — pick from 9 built-in premium voices, with optional style control (emotion, pacing, tone).

All processing happens **locally on your Mac** — nothing is uploaded anywhere.

---

## ✅ Before you start (checklist)

You'll need all of these. Instructions for each are below.

| Requirement | Why |
|---|---|
| A Mac with **Apple Silicon** (M1, M2, M3, M4, M5 or so on...) | Required — MLX doesn't run on Intel Macs |
| **Homebrew** | Makes installing the tools below one-line easy |
| **Python** 3.10+ | Runs the app |
| **Git** | Downloads (clones) this repository |
| **FFmpeg** | Lets the app read/convert audio files |

### 1. Confirm you have Apple Silicon

Click the Apple menu (**) → **About This Mac**. Under "Chip," you should see something like *Apple M1*, *M2*, *M3*, or *M4*. If it says *Intel*, this repo unfortunately won't work on your Mac — use the NVIDIA repo on a Windows/Linux PC with an NVIDIA GPU instead.

You can also check this in Terminal:
```bash
uname -m
```
This should print `arm64` (not `x86_64`).

### 2. Install Homebrew (if you don't already have it)

Open the **Terminal** app (search for it with Spotlight: `Cmd + Space`, then type "Terminal"), and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Follow any on-screen instructions it gives you at the end (it may ask you to run one or two extra commands to finish setup — copy-paste exactly what it shows you).

Check it worked:
```bash
brew --version
```

### 3. Install Python, Git, and FFmpeg via Homebrew

```bash
brew install python git ffmpeg
```

Then verify each one:
```bash
python3 --version
git --version
ffmpeg -version
```
All three should print version info without errors.

---

## 🚀 Installation

**1. Clone this repository:**
```bash
git clone https://github.com/samiu11ah/qwen3-tts-apple-silicon-demo.git
cd qwen3-tts-apple-silicon-demo
```

**2. Create a virtual environment** (keeps this app's dependencies separate from everything else on your Mac):
```bash
python3 -m venv venv
```

**3. Activate it:**
```bash
source venv/bin/activate
```
You'll know it worked because your terminal prompt now starts with `(venv)`.

**4. Install the required packages:**
```bash
pip install -r requirements.txt
```
This may take a few minutes.

**5. Run the app:**
```bash
python mlx_app.py
```

**6. Open the app:** the terminal will print a local web address, usually:
```
http://127.0.0.1:7860
```
Open that link in your browser (Safari, Chrome, etc.) — the app should load with three tabs: Clone, VoiceDesign, CustomVoice.

> ⏳ **First run only:** the first time you use each tab, the app automatically downloads the matching AI model (roughly 2.5 GB for the 0.6B models, 4.5 GB for the 1.7B models). This can take a few minutes depending on your internet speed — subsequent runs are instant since the model is cached on your disk.

---

## 🧯 Troubleshooting

**Repeated/garbled words when cloning a voice**
Make sure the **Reference transcript** field exactly matches what's spoken in your uploaded reference audio clip — this model needs an accurate transcript to align properly, and a missing or wrong transcript is the most common cause of stuttering or repeated words.

**"Model weights couldn't be found or downloaded"**
Check your internet connection. If you're behind a restrictive firewall, Hugging Face downloads may be blocked — try a different network.

**"Ran out of memory"**
Switch to the **0.6B** model size in the dropdown, close other memory-heavy apps, or shorten your input text.

**`zsh: command not found: python` or `pip`**
Make sure your virtual environment is activated (`source venv/bin/activate` — your prompt should show `(venv)`), and that Homebrew's Python was actually installed (`brew install python`).

**Port 7860 already in use**
Close any other Gradio app that might already be running, or edit the last line of `mlx_app.py` to `demo.queue().launch(server_port=7861)` (or any free port number).

**Something else feels off after a Mac restart**
Re-activate your virtual environment (`source venv/bin/activate`) before running `python mlx_app.py` again — it doesn't stay active between terminal sessions.

---

## 📁 What's in this repo

| File | Purpose |
|---|---|
| `mlx_app.py` | The Gradio app — run this to start |
| `requirements.txt` | Python packages needed to run the app |
| `README.md` | This guide |
| `LICENSE` | This repo's open-source license |

---

## 🙏 Credits & Licenses

- Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba, released under the **Apache License 2.0**, and the [mlx-audio](https://github.com/Blaizzy/mlx-audio) project for Apple Silicon inference. Model weights are downloaded automatically from Hugging Face (via the `mlx-community` mirrors) and are governed by that same Apache 2.0 license — see the [model cards](https://huggingface.co/collections/Qwen/qwen3-tts) for details.
- The code in **this repository** (the Gradio app itself) is released under the **MIT License** — see [`LICENSE`](LICENSE).

## ⚠️ Responsible use

Voice cloning is powerful — only clone voices you have permission to use (your own voice, or someone who has explicitly consented). Don't use this to impersonate real people without their consent, or to create misleading or deceptive content.
