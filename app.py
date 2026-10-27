import os
import io
import time
import json
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models
from gtts import gTTS

# -----------------------------------------------------------------------------
# Page Configuration & Accessibility Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Caption Generator for Visually Impaired",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom accessibility-friendly styling: high contrast, clear fonts, elegant layout
st.markdown("""
<style>
    .project-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: #FFFFFF;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
    }
    .project-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #FFFFFF;
    }
    .project-tagline {
        font-size: 1.05rem;
        color: #E0E7FF;
        font-weight: 400;
    }
    .assistive-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .caption-card {
        background: #F8FAFC;
        border-left: 6px solid #2563EB;
        padding: 1.4rem 1.6rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #0F172A;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .meta-box {
        display: flex;
        gap: 15px;
        margin-bottom: 1rem;
    }
    .metric-pill {
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Device Configuration
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Model Architecture (ResNet-50 Encoder + LSTM Decoder)
# -----------------------------------------------------------------------------
class CaptionModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512):
        super().__init__()
        self.feature_projection = nn.Linear(2048, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        """Original forward pass used during Flickr8k training."""
        image_embedding = self.feature_projection(image_features).unsqueeze(1)
        word_embeddings = self.embedding(captions)
        inputs = torch.cat([image_embedding, word_embeddings[:, :-1, :]], dim=1)
        output, _ = self.lstm(inputs)
        return self.fc(output)

    def generate_greedy(self, image_features, word_to_idx, idx_to_word, max_length=25):
        """Greedy recurrent generation starting from ResNet-50 visual representation."""
        with torch.no_grad():
            image_emb = self.feature_projection(image_features).unsqueeze(1)
            # Step 0: Image feature initializes the LSTM hidden state
            _, (h, c) = self.lstm(image_emb)

            start_id = word_to_idx.get("<start>", 1)
            end_id = word_to_idx.get("<end>", 2)
            pad_id = word_to_idx.get("<pad>", 0)
            unk_id = word_to_idx.get("<unk>", 3)

            curr_token = torch.tensor([[start_id]], device=image_features.device)
            words = []

            for _ in range(max_length):
                emb = self.embedding(curr_token)
                lstm_out, (h, c) = self.lstm(emb, (h, c))
                logits = self.fc(lstm_out)
                next_id = logits[0, -1].argmax().item()

                if next_id in (end_id, pad_id):
                    break
                if next_id not in (start_id, unk_id):
                    word = idx_to_word.get(next_id, "")
                    if word:
                        words.append(word)

                curr_token = torch.tensor([[next_id]], device=image_features.device)

            return " ".join(words)

    def generate_beam_search(self, image_features, word_to_idx, idx_to_word, beam_width=3, max_length=25):
        """Beam search decoding for multi-path exploration and descriptive coherence."""
        with torch.no_grad():
            image_emb = self.feature_projection(image_features).unsqueeze(1)
            _, (h, c) = self.lstm(image_emb)

            start_id = word_to_idx.get("<start>", 1)
            end_id = word_to_idx.get("<end>", 2)
            pad_id = word_to_idx.get("<pad>", 0)
            unk_id = word_to_idx.get("<unk>", 3)

            beams = [(0.0, [start_id], (h, c))]
            completed = []

            for _ in range(max_length):
                candidates = []
                for score, tokens, state in beams:
                    last_token = torch.tensor([[tokens[-1]]], device=image_features.device)
                    emb = self.embedding(last_token)
                    out, new_state = self.lstm(emb, state)
                    logits = self.fc(out)[0, -1]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_probs, topk_ids = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        cand_id = topk_ids[i].item()
                        cand_score = score + topk_probs[i].item()

                        if cand_id == end_id:
                            # Normalize by sequence length penalty
                            norm_score = cand_score / (len(tokens) ** 0.7)
                            completed.append((norm_score, tokens + [cand_id]))
                        else:
                            candidates.append((cand_score, tokens + [cand_id], new_state))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]
                if not beams:
                    break

            best_tokens = completed[0][1] if completed else beams[0][1]
            special = {start_id, end_id, pad_id, unk_id}
            words = [idx_to_word.get(tid, "") for tid in best_tokens if tid not in special]
            return " ".join([w for w in words if w])

# -----------------------------------------------------------------------------
# Component Loader (Cached for fast subsequent inferences)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing ResNet-50 & LSTM Model...")
def load_components():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, "vocabulary.json")
    model_path = os.path.join(base_dir, "caption_model.pth")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file missing at {vocab_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at {model_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(v): k for k, v in word_to_idx.items()}

    # Load ResNet-50 feature extractor
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    encoder = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

    # Load LSTM Caption Model
    model = CaptionModel(vocab_size=len(word_to_idx)).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return encoder, model, weights.transforms(), word_to_idx, idx_to_word

try:
    encoder, model, preprocess, word_to_idx, idx_to_word = load_components()
except Exception as e:
    st.error(f"❌ Error loading model components: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: Controls & Assistive Options
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Project Controls")

    st.subheader("🔍 Decoding Strategy")
    decoding_method = st.radio(
        "Algorithm:",
        options=["Greedy Search (Fast & Accurate)", "Beam Search (Multi-Candidate)"],
        index=0,
        help="Greedy Search picks the highest probability word at each step. Beam Search explores multiple sequence paths."
    )

    beam_width = 3
    if "Beam Search" in decoding_method:
        beam_width = st.slider("Beam Width", min_value=2, max_value=7, value=3, step=1)

    max_length = st.slider("Maximum Sentence Length", min_value=10, max_value=40, value=25, step=5)

    st.divider()
    st.subheader("🔊 Voice Narration (Assistive)")
    enable_voice = st.checkbox("Enable Audio Narration", value=True)
    auto_play_voice = st.checkbox("Auto-Play Narration", value=True, help="Automatically play spoken audio as soon as caption is generated")

    accent_map = {
        "English (US)": "com",
        "English (UK)": "co.uk",
        "English (India)": "co.in",
        "English (Australia)": "com.au",
        "English (Canada)": "ca"
    }
    selected_accent = st.selectbox("Voice Accent:", options=list(accent_map.keys()), index=0)
    tld_val = accent_map[selected_accent]

    st.divider()
    st.subheader("📊 College Project Details")
    st.markdown(f"**Architecture:** `ResNet-50 + LSTM`")
    st.markdown(f"**Dataset:** `Flickr8k (8,000 photos)`")
    st.markdown(f"**Vocabulary:** `{len(word_to_idx):,} tokens`")
    st.markdown(f"**Inference Compute:** `{device.type.upper()}`")

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown("""
<div class='project-header'>
    <div class='project-title'>👁️ Image Caption Generator with Voice Output</div>
    <div class='project-tagline'>An Assistive Deep Learning System for the Visually Impaired</div>
    <div class='assistive-badge'>🎓 College Demonstration & Evaluation Ready</div>
</div>
""", unsafe_allow_html=True)

# Image Input Methods
input_mode = st.radio(
    "Choose Image Input Method:",
    options=["📁 Upload Image", "🖼️ Test with Demo Images", "📷 Take Photo with Webcam"],
    horizontal=True
)

image = None
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Select a photo from your computer (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")

elif input_mode == "🖼️ Test with Demo Images":
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if sample_files:
        selected_sample = st.selectbox("Select a demo image:", sample_files)
        image_path = os.path.join(sample_dir, selected_sample)
        image = Image.open(image_path).convert("RGB")
    else:
        st.info("Demo images directory is empty. Please upload an image.")

elif input_mode == "📷 Take Photo with Webcam":
    cam_file = st.camera_input("Capture an image using your camera")
    if cam_file is not None:
        image = Image.open(cam_file).convert("RGB")

# -----------------------------------------------------------------------------
# Caption & Speech Generation Processing
# -----------------------------------------------------------------------------
if image is not None:
    col_img, col_out = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("📷 Image Preview")
        st.image(image, use_container_width=True)
        st.caption(f"Resolution: {image.width} × {image.height} px | Color: {image.mode}")

    with col_out:
        st.subheader("💬 Generated Description & Voice")
        
        btn_generate = st.button("✨ Generate Caption & Speak", type="primary", use_container_width=True)

        if btn_generate:
            start_time = time.time()
            with st.spinner("Extracting ResNet-50 features & predicting tokens..."):
                # 1. Preprocess & extract 2048-dim feature vector
                tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = encoder(tensor).flatten(1)

                # 2. Generate caption with chosen algorithm
                if "Beam Search" in decoding_method:
                    caption_raw = model.generate_beam_search(
                        features, word_to_idx, idx_to_word, beam_width=beam_width, max_length=max_length
                    )
                else:
                    caption_raw = model.generate_greedy(
                        features, word_to_idx, idx_to_word, max_length=max_length
                    )

            elapsed_ms = (time.time() - start_time) * 1000

            if caption_raw and caption_raw.strip():
                # Format into a clean, capitalized English sentence
                final_caption = caption_raw.strip().capitalize()
                if not final_caption.endswith(('.', '!', '?')):
                    final_caption += '.'

                # Display high-contrast accessible caption card
                st.markdown(f"<div class='caption-card'>\"{final_caption}\"</div>", unsafe_allow_html=True)

                # Metrics row
                word_count = len(final_caption.split())
                st.markdown(f"""
                <div class='meta-box'>
                    <span class='metric-pill'>⏱️ Inference: {elapsed_ms:.1f} ms</span>
                    <span class='metric-pill'>📝 Words: {word_count}</span>
                    <span class='metric-pill'>💻 {device.type.upper()}</span>
                </div>
                """, unsafe_allow_html=True)

                # Text box for easy copy/paste
                st.text_input("Copy Caption Text:", value=final_caption, label_visibility="collapsed")

                # Voice output generation
                if enable_voice:
                    try:
                        with st.spinner("Synthesizing voice narration with Google TTS..."):
                            audio_buffer = io.BytesIO()
                            tts = gTTS(text=final_caption, lang="en", tld=tld_val)
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            audio_bytes = audio_buffer.read()

                        st.markdown("**🔊 Spoken Description (Audio):**")
                        st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play_voice)

                        # Download button for assistive MP3 devices
                        st.download_button(
                            label="⬇️ Download Voice File (.mp3)",
                            data=audio_bytes,
                            file_name="scene_description.mp3",
                            mime="audio/mp3",
                            use_container_width=True
                        )
                    except Exception as audio_err:
                        st.warning(f"Voice generation unavailable: {audio_err}. Ensure an active internet connection for Google TTS.")
            else:
                st.warning("Could not generate a descriptive caption for this image. Please try another image.")

st.divider()

# -----------------------------------------------------------------------------
# Technical Project Information (for College Evaluation)
# -----------------------------------------------------------------------------
with st.expander("🎓 Project Technical Details & Viva Questions Guide"):
    st.markdown("""
    ### 1. Problem Statement & Motivation
    - **Target Users:** Visually impaired and low-vision individuals.
    - **Solution:** An end-to-end multimodal deep learning pipeline that perceives the surrounding visual environment through cameras/photos, synthesizes natural language descriptions, and narrates them aloud in real-time.

    ### 2. Deep Learning Pipeline
    1. **Encoder (ResNet-50 CNN):**
       - A deep 50-layer Residual Network pre-trained on ImageNet.
       - The final classification layer is removed to extract the **2048-dimensional** bottleneck feature representation.
    2. **Linear Feature Projection:**
       - Compresses the 2048-dim visual feature vector into a **256-dim** latent space matching the word embedding size.
    3. **Decoder (LSTM Network):**
       - Recurrent neural network with an embedding dimension of **256** and hidden size of **512**.
       - Conditioned initially on the projected image feature vector to seed the hidden state, followed sequentially by word tokens until `<end>` is produced.
    4. **Vocabulary & Dataset:**
       - Trained on the benchmark **Flickr8k dataset** (8,000 photos with 5 human reference captions per photo = 40,000 captions).
       - Vocabulary contains **8,782** unique tokens.

    ### 3. Key Viva / Evaluation Questions & Answers
    - **Q: Why use ResNet-50 instead of standard CNNs?**
      - *A:* ResNet introduces skip/residual connections that resolve the vanishing gradient problem, allowing much deeper feature extraction without degradation.
    - **Q: Why LSTM instead of vanilla RNN?**
      - *A:* Standard RNNs suffer from vanishing and exploding gradients over sequences. LSTMs utilize gating mechanisms (forget gate, input gate, output gate) to remember context over long captions.
    - **Q: What is the purpose of the `<start>` and `<end>` tokens?**
      - *A:* The `<start>` token initiates generation given the image context, and `<end>` instructs the network when the sentence is logically complete so it stops decoding.
    """)
