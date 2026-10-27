"""Unified Streamlit UI for GenAI lab Experiments 1–10.

Run from the project directory with: streamlit run app.py
"""
from __future__ import annotations
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from experiments.exp01_dataset import analyze_dataset, summarize_dataset_with_langchain
from experiments.exp02_tensorflow import run_tensorflow_demo
from experiments.exp03_qa import load_qa_pipeline, answer_question
from experiments.exp04_image import load_image_pipeline, generate_image
from experiments.exp05_vae import train_vae
from experiments.exp06_dcgan import train_dcgan
from experiments.exp07_stylegan import train_stylegan
from experiments.exp08_music import load_music_pipeline, generate_music
from experiments.exp09_rag_lora import (
    extract_document_text, chunk_text, load_embedding_model,
    answer_base_and_rag, parse_qa_examples, fine_tune_lora, answer_with_lora,
)
from experiments.exp10_video import load_video_pipeline, generate_video

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="GenAI Lab Experiments 1–10", page_icon="🧪", layout="wide")
st.title("GenAI Lab Experiments — Unified Workbench")
st.caption("Experiments 1–10 · Python · TensorFlow · PyTorch · Hugging Face · RAG · Generative AI")
st.info("First-time runs for pretrained models download model weights. For SDXL, DCGAN, StyleGAN, VAE training, MusicGen, LoRA and LTX-Video, use a suitable GPU runtime where possible.")

EXPERIMENTS = [
    "Experiment 1 — Dataset analysis & preprocessing",
    "Experiment 2 — Python + TensorFlow tensor operations",
    "Experiment 3 — FLAN-T5 question-answer system",
    "Experiment 4 — Text-to-image (Stable Diffusion XL)",
    "Experiment 5 — MNIST Variational Autoencoder (VAE)",
    "Experiment 6 — DCGAN face generation (CelebA-style images)",
    "Experiment 7 — StyleGAN-inspired face generation",
    "Experiment 8 — MusicGen text-to-music",
    "Experiment 9 — Personalized assistant (RAG + optional LoRA)",
    "Experiment 10 — LTX-Video text-to-video",
]
selected = st.sidebar.selectbox("Choose experiment", EXPERIMENTS)
st.sidebar.markdown("---")
st.sidebar.markdown("**Source note**")
st.sidebar.caption("The uploaded TensorFlow tensor-operations manual is titled Experiment No. 1 internally but was uploaded as Experiment No. 2. It is listed here as Experiment 2 to avoid the filename conflict.")
st.sidebar.markdown("---")
st.sidebar.caption("Outputs are stored in the project's `outputs/` folder.")


@st.cache_resource(show_spinner="Loading the pretrained FLAN-T5 model…")
def cached_qa_pipeline():
    return load_qa_pipeline()


@st.cache_resource(show_spinner="Loading the SDXL image-generation model…")
def cached_image_pipeline():
    return load_image_pipeline()


@st.cache_resource(show_spinner="Loading the MusicGen model…")
def cached_music_pipeline():
    return load_music_pipeline()


@st.cache_resource(show_spinner="Loading the sentence embedding model…")
def cached_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Loading the LTX-Video model…")
def cached_video_pipeline():
    return load_video_pipeline()


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Use a stable basename only; no uploaded directory path is trusted.
    safe_name = Path(uploaded_file.name).name.replace("..", "_")
    target = dest_dir / safe_name
    target.write_bytes(uploaded_file.getvalue())
    return target


def safe_extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract a user ZIP only after rejecting members that escape the destination."""
    extract_to.mkdir(parents=True, exist_ok=True)
    destination = extract_to.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(destination)
    return destination


def display_image_paths(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            st.image(str(p), caption=p.name, use_container_width=True)


def make_download_button(path: str | Path, label: str, key: str):
    p = Path(path)
    if p.exists():
        st.download_button(label, data=p.read_bytes(), file_name=p.name, mime="application/octet-stream", key=key)


# ---------------------------------------------------------------------------
# Experiment 1 — Dataset analysis / preprocessing
# ---------------------------------------------------------------------------
if selected.startswith("Experiment 1"):
    st.header("Experiment 1 — Comprehensive dataset analysis and preprocessing")
    st.write("Upload a CSV or Excel dataset. The app inspects dimensions, types, summary statistics, missing values and duplicates, then exports a cleaned CSV and visualizations.")
    uploaded = st.file_uploader("Choose dataset", type=["csv", "xlsx", "xls"], key="exp01_upload")
    if st.button("Analyze and preprocess dataset", type="primary", disabled=uploaded is None):
        try:
            run_dir = OUTPUTS / "exp01"
            source_dir = run_dir / "uploads"
            uploaded_path = save_uploaded_file(uploaded, source_dir)
            with st.spinner("Analyzing and cleaning dataset…"):
                result = analyze_dataset(uploaded_path, run_dir)
            st.session_state["exp01_result"] = result
        except Exception as exc:
            st.error(f"Experiment 1 failed: {exc}")
    result = st.session_state.get("exp01_result")
    if result:
        report = result["report"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original rows", report["original_rows"])
        col2.metric("Cleaned rows", report["cleaned_rows"])
        col3.metric("Columns", report["original_columns"])
        col4.metric("Duplicates removed", report["removed_duplicate_rows"])
        st.subheader("Dataset preview")
        st.dataframe(result["dataframe"].head(20), use_container_width=True)
        st.subheader("Cleaned dataset preview")
        st.dataframe(result["cleaned_dataframe"].head(20), use_container_width=True)
        with st.expander("Full report"):
            st.json(report)
        st.download_button("Download cleaned CSV", result["cleaned_dataframe"].to_csv(index=False).encode("utf-8"), "cleaned_dataset.csv", "text/csv")
        st.download_button("Download JSON report", json.dumps(report, indent=2, default=str).encode("utf-8"), "dataset_report.json", "application/json")
        st.caption(f"ML-ready feature matrix shape: {report.get('preprocessed_feature_shape')}")
        make_download_button(result["features_path"], "Download preprocessed_features.npy", "exp01_features")
        st.subheader("Visualizations")
        display_image_paths(result["plots"])
        with st.expander("Optional: LangChain dataset summary (local FLAN-T5 model)"):
            st.write("This uses LangChain's Hugging Face integration. The first run downloads FLAN-T5 weights; no external API key is required.")
            if st.button("Generate LangChain summary", key="exp01_langchain_summary"):
                try:
                    with st.spinner("Generating dataset narrative with LangChain…"):
                        st.session_state["exp01_langchain_summary"] = summarize_dataset_with_langchain(result["dataframe"], cached_qa_pipeline())
                except Exception as exc:
                    st.error(f"LangChain summary failed: {exc}. Check the langchain-core and langchain-huggingface dependencies.")
            if st.session_state.get("exp01_langchain_summary"):
                st.write(st.session_state["exp01_langchain_summary"])

# ---------------------------------------------------------------------------
# Experiment 2 — TensorFlow operations
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 2"):
    st.header("Experiment 2 — Basic Python and TensorFlow 2 tensor operations")
    st.write("Creates scalar, vector, matrix and 3-D tensors, and demonstrates properties, arithmetic, mathematical functions, reshape and NumPy conversion.")
    if st.button("Run TensorFlow tensor demo", type="primary"):
        try:
            with st.spinner("Running TensorFlow operations…"):
                st.session_state["exp02_result"] = run_tensorflow_demo()
        except Exception as exc:
            st.error(f"Experiment 2 failed. Check that TensorFlow is installed: {exc}")
    result = st.session_state.get("exp02_result")
    if result:
        st.success(f"TensorFlow {result['tensorflow_version']} · NumPy {result['numpy_version']}")
        st.subheader("Basic Python operations")
        st.json(result["python_demo"])
        st.subheader("Tensor properties")
        for name, item in result["tensors"].items():
            with st.expander(f"{name} — shape {item['shape']}, rank {item['rank']}, dtype {item['dtype']}"):
                st.json(item)
        st.subheader("Arithmetic and tensor functions")
        st.json(result["operations"])

# ---------------------------------------------------------------------------
# Experiment 3 — FLAN-T5 QA
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 3"):
    st.header("Experiment 3 — Text generation using pretrained FLAN-T5")
    st.write("Enter a question. FLAN-T5 generates a natural-language answer through a pretrained Transformer model.")
    question = st.text_area("Your question", "What is a Transformer model in Generative AI?", key="exp03_question")
    max_tokens = st.slider("Maximum generated tokens", 32, 256, 160, key="exp03_tokens")
    if st.button("Generate answer", type="primary"):
        try:
            pipe = cached_qa_pipeline()
            with st.spinner("Generating answer…"):
                answer = answer_question(question, pipe, max_new_tokens=max_tokens)
            st.session_state["exp03_answer"] = answer
        except Exception as exc:
            st.error(f"Question-answer generation failed: {exc}")
    if st.session_state.get("exp03_answer"):
        st.subheader("Generated answer")
        st.write(st.session_state["exp03_answer"])

# ---------------------------------------------------------------------------
# Experiment 4 — SDXL image generation
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 4"):
    st.header("Experiment 4 — Text-to-image generation using Stable Diffusion XL")
    prompt = st.text_area("Image prompt", "A small wooden boat on a calm lake at sunrise, mist over the water, cinematic landscape photography, highly detailed", key="exp04_prompt")
    negative = st.text_input("Negative prompt", "blurry, low quality, distorted, artifacts", key="exp04_negative")
    col1, col2 = st.columns(2)
    steps = col1.slider("Inference steps", 10, 50, 25, key="exp04_steps")
    guidance = col2.slider("Guidance scale", 1.0, 12.0, 7.0, 0.5, key="exp04_guidance")
    st.caption("SDXL's base model is large. A CUDA GPU with sufficient VRAM is recommended; the model is loaded using CPU offload when available.")
    if st.button("Generate image", type="primary"):
        try:
            pipe = cached_image_pipeline()
            with st.spinner("Generating image… this may take a few minutes on the first run."):
                path = generate_image(prompt, pipe, OUTPUTS / "exp04" / "generated_image.png", steps, guidance, negative)
            st.session_state["exp04_image_path"] = path
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")
    image_path = st.session_state.get("exp04_image_path")
    if image_path and Path(image_path).exists():
        st.image(image_path, caption="Generated image", use_container_width=True)
        make_download_button(image_path, "Download PNG", "exp04_download")

# ---------------------------------------------------------------------------
# Experiment 5 — VAE
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 5"):
    st.header("Experiment 5 — Variational Autoencoder on MNIST")
    st.write("Trains an encoder, a 2-D probabilistic latent space, and a decoder. It exports loss curves, reconstructions, latent-space visualization, and newly generated digits.")
    col1, col2 = st.columns(2)
    epochs = col1.slider("Training epochs (manual suggests 30)", 1, 30, 3, key="exp05_epochs")
    batch_size = col2.selectbox("Batch size", [32, 64, 128, 256], index=2, key="exp05_batch")
    latent_dim = st.selectbox("Latent dimension", [2, 4, 8], index=0, key="exp05_latent")
    if st.button("Train VAE and generate digits", type="primary"):
        try:
            with st.spinner("Downloading MNIST (first run) and training the VAE…"):
                st.session_state["exp05_result"] = train_vae(epochs, batch_size, latent_dim, OUTPUTS / "exp05")
        except Exception as exc:
            st.error(f"VAE training failed: {exc}")
    result = st.session_state.get("exp05_result")
    if result:
        st.success("VAE training and generation completed.")
        st.json(result["history"])
        display_image_paths(result["plots"])
        make_download_button(result["encoder_path"], "Download encoder.keras", "exp05_encoder")
        make_download_button(result["decoder_path"], "Download decoder.keras", "exp05_decoder")

# ---------------------------------------------------------------------------
# Experiment 6 — DCGAN
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 6"):
    st.header("Experiment 6 — Face generation using DCGAN")
    st.write("Upload a ZIP containing CelebA-style face images (JPG/PNG/WebP). The training code uses 64×64 RGB images, a 100-dimensional noise vector, a Generator and a Discriminator.")
    zip_upload = st.file_uploader("Upload face-image ZIP", type=["zip"], key="exp06_zip")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 1, 20, 1, key="exp06_epochs")
    batch_size = col2.selectbox("Batch size", [8, 16, 32, 64], index=1, key="exp06_batch")
    max_batches = col3.slider("Max batches per epoch", 1, 500, 40, key="exp06_batches")
    if st.button("Train DCGAN", type="primary", disabled=zip_upload is None):
        try:
            run_dir = OUTPUTS / "exp06"
            uploaded_zip = save_uploaded_file(zip_upload, run_dir / "uploads")
            data_dir = safe_extract_zip(uploaded_zip, run_dir / "dataset")
            with st.spinner("Training DCGAN… the quality of the generated faces improves with more data and training."):
                st.session_state["exp06_result"] = train_dcgan(data_dir, epochs, batch_size, max_batches, run_dir)
        except Exception as exc:
            st.error(f"DCGAN training failed: {exc}")
    result = st.session_state.get("exp06_result")
    if result:
        st.write(f"Device: `{result['device']}` · Training images: `{result['training_images']}`")
        display_image_paths([result["generated_grid_path"], result["loss_plot_path"]])
        make_download_button(result["generator_path"], "Download generator.pth", "exp06_gen")
        make_download_button(result["discriminator_path"], "Download discriminator.pth", "exp06_disc")

# ---------------------------------------------------------------------------
# Experiment 7 — StyleGAN-inspired
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 7"):
    st.header("Experiment 7 — StyleGAN-inspired image generation")
    st.warning("This is a compact educational StyleGAN-inspired implementation that demonstrates mapping, layer-wise styles, style mixing and noise injection. It is not the full NVIDIA StyleGAN2-ADA implementation; short lab training will not generate publication-quality faces.")
    st.write("Upload a ZIP of face images. The module trains a small generator and saves a style-mixing/noise comparison grid.")
    zip_upload = st.file_uploader("Upload face-image ZIP", type=["zip"], key="exp07_zip")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 1, 20, 1, key="exp07_epochs")
    batch_size = col2.selectbox("Batch size", [4, 8, 16, 32], index=2, key="exp07_batch")
    max_batches = col3.slider("Max batches per epoch", 1, 300, 40, key="exp07_batches")
    if st.button("Train StyleGAN-inspired model", type="primary", disabled=zip_upload is None):
        try:
            run_dir = OUTPUTS / "exp07"
            uploaded_zip = save_uploaded_file(zip_upload, run_dir / "uploads")
            data_dir = safe_extract_zip(uploaded_zip, run_dir / "dataset")
            with st.spinner("Training style-based GAN…"):
                st.session_state["exp07_result"] = train_stylegan(data_dir, epochs, batch_size, max_batches, run_dir)
        except Exception as exc:
            st.error(f"StyleGAN experiment failed: {exc}")
    result = st.session_state.get("exp07_result")
    if result:
        st.write(f"Device: `{result['device']}` · Training images: `{result['training_images']}`")
        display_image_paths([result["generated_grid_path"], result["style_mixing_noise_path"], result["loss_plot_path"]])
        make_download_button(result["generator_path"], "Download style_generator.pth", "exp07_gen")

# ---------------------------------------------------------------------------
# Experiment 8 — MusicGen
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 8"):
    st.header("Experiment 8 — Music generation with pretrained MusicGen")
    prompt = st.text_area("Music prompt", "Relaxing instrumental music with piano, soft flute, gentle ambient textures and a peaceful atmosphere", key="exp08_prompt")
    col1, col2 = st.columns(2)
    duration = col1.slider("Duration (seconds, maximum 30)", 3, 20, 6, key="exp08_duration")
    temperature = col2.slider("Sampling temperature", 0.1, 1.5, 0.9, 0.1, key="exp08_temp")
    st.caption("The first run downloads the MusicGen-small checkpoint. A GPU is strongly recommended; CPU generation can be slow.")
    if st.button("Generate music", type="primary"):
        try:
            pipe = cached_music_pipeline()
            with st.spinner("Generating music…"):
                result = generate_music(prompt, pipe, duration, OUTPUTS / "exp08" / "generated_music.wav", temperature)
            st.session_state["exp08_result"] = result
        except Exception as exc:
            st.error(f"Music generation failed: {exc}")
    result = st.session_state.get("exp08_result")
    if result and Path(result["audio_path"]).exists():
        st.success(f"Generated about {result['duration_seconds']:.1f} seconds at {result['sample_rate']} Hz.")
        st.audio(Path(result["audio_path"]).read_bytes(), format="audio/wav")
        make_download_button(result["audio_path"], "Download generated_music.wav", "exp08_download")

# ---------------------------------------------------------------------------
# Experiment 9 — RAG + LoRA
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 9"):
    st.header("Experiment 9 — Personalized assistant using RAG and optional LoRA")
    st.write("Upload a TXT, PDF, DOCX or Markdown document. The RAG path chunks the document, embeds chunks, retrieves relevant passages, and compares a base answer with a context-grounded answer.")
    doc_upload = st.file_uploader("Upload a knowledge document", type=["txt", "pdf", "docx", "md", "markdown"], key="exp09_doc")
    question = st.text_area("Ask a question about the document", "What are the main points in this document?", key="exp09_question")
    top_k = st.slider("Retrieved chunks", 1, 6, 3, key="exp09_topk")
    if st.button("Compare base LLM and RAG answers", type="primary", disabled=doc_upload is None):
        try:
            doc_dir = OUTPUTS / "exp09" / "documents"
            doc_path = save_uploaded_file(doc_upload, doc_dir)
            with st.spinner("Reading document and retrieving relevant passages…"):
                document_text = extract_document_text(doc_path)
                if not document_text.strip():
                    raise ValueError("No text could be extracted from this file. Scanned PDFs may require OCR.")
                chunks = chunk_text(document_text)
                embedder = cached_embedding_model()
                qa_pipe = cached_qa_pipeline()
                result = answer_base_and_rag(question, chunks, embedder, qa_pipe, top_k=top_k)
            st.session_state["exp09_rag_result"] = result
        except Exception as exc:
            st.error(f"RAG assistant failed: {exc}")
    result = st.session_state.get("exp09_rag_result")
    if result:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Base model answer")
            st.write(result["base_answer"])
        with col2:
            st.subheader("RAG-grounded answer")
            st.write(result["rag_answer"])
        st.subheader("Retrieved document excerpts")
        for i, item in enumerate(result["retrieved"], 1):
            with st.expander(f"Excerpt {i} · similarity {item['score']:.3f}"):
                st.write(item["chunk"])

    st.markdown("---")
    st.subheader("Optional: LoRA / QLoRA fine-tuning")
    st.write("Enter small example pairs with one question and answer per line, separated by a TAB. This trains a small Qwen 0.5B adapter; it is a separate demonstration from RAG.")
    default_pairs = "What is RAG?\tRAG retrieves relevant external text and provides it as context to a language model.\nWhat is LoRA?\tLoRA fine-tunes a small set of adapter parameters instead of updating every parameter of the base model."
    pairs_text = st.text_area("Training pairs (question<TAB>answer per line)", default_pairs, height=120, key="exp09_pairs")
    col1, col2, col3 = st.columns(3)
    lora_epochs = col1.slider("LoRA epochs", 1, 5, 1, key="exp09_lora_epochs")
    use_4bit = col2.checkbox("Use QLoRA 4-bit (CUDA + bitsandbytes required)", value=False, key="exp09_4bit")
    lora_question = col3.text_input("Test question after training", "What is LoRA?", key="exp09_lora_test")
    if st.button("Train LoRA adapter", type="secondary"):
        try:
            examples = parse_qa_examples(pairs_text)
            with st.spinner("Downloading the base LLM and fine-tuning the LoRA adapter…"):
                result = fine_tune_lora(examples, OUTPUTS / "exp09" / "lora_adapter", epochs=lora_epochs, use_4bit=use_4bit)
            st.session_state["exp09_lora_result"] = result
        except Exception as exc:
            st.error(f"LoRA fine-tuning failed: {exc}")
    lora_result = st.session_state.get("exp09_lora_result")
    if lora_result:
        st.success(f"LoRA adapter saved to {lora_result['adapter_dir']}")
        st.json(lora_result)
        if st.button("Ask fine-tuned model", key="exp09_ask_finetuned"):
            try:
                with st.spinner("Loading the base model plus LoRA adapter…"):
                    answer = answer_with_lora(lora_question, lora_result["adapter_dir"])
                st.session_state["exp09_lora_answer"] = answer
            except Exception as exc:
                st.error(f"Fine-tuned inference failed: {exc}")
    if st.session_state.get("exp09_lora_answer"):
        st.subheader("Fine-tuned model answer")
        st.write(st.session_state["exp09_lora_answer"])

# ---------------------------------------------------------------------------
# Experiment 10 — LTX-Video
# ---------------------------------------------------------------------------
elif selected.startswith("Experiment 10"):
    st.header("Experiment 10 — Text-to-video generation using LTX-Video")
    prompt = st.text_area("Video prompt", "A small wooden boat sailing across a calm lake at sunrise, gentle water movement, mist over the water, golden sunlight, cinematic wide shot", key="exp10_prompt")
    negative = st.text_input("Negative prompt", "worst quality, blurry, inconsistent motion, jittery, distorted", key="exp10_negative")
    col1, col2, col3 = st.columns(3)
    resolution = col1.selectbox("Resolution", ["512 × 320", "768 × 512"], index=0, key="exp10_resolution")
    frames = col2.selectbox("Frames (must be 8n+1)", [17, 25, 33, 49], index=1, key="exp10_frames")
    steps = col3.slider("Inference steps", 10, 50, 30, key="exp10_steps")
    st.caption("Requires a CUDA GPU; the documented LTX-Video configuration uses about 10 GB VRAM. The first run downloads model weights.")
    if st.button("Generate video", type="primary"):
        try:
            pipe = cached_video_pipeline()
            width, height = (512, 320) if resolution.startswith("512") else (768, 512)
            with st.spinner("Generating video… this may take several minutes."):
                result = generate_video(prompt, pipe, OUTPUTS / "exp10" / "generated_video.mp4", width, height, frames, steps, negative_prompt=negative)
            st.session_state["exp10_result"] = result
        except Exception as exc:
            st.error(f"Video generation failed: {exc}")
    result = st.session_state.get("exp10_result")
    if result and Path(result["video_path"]).exists():
        st.video(result["video_path"])
        st.write(f"Frames: {result['num_frames']} · FPS: {result['fps']} · Approx. duration: {result['duration_seconds']:.2f}s")
        make_download_button(result["video_path"], "Download generated_video.mp4", "exp10_download")

# Source browser: the unified app also lets students inspect/download each experiment's code.
with st.expander("Browse or download the Python source code", expanded=False):
    source_map = {
        "Experiment 1 — dataset analysis": ROOT / "experiments" / "exp01_dataset.py",
        "Experiment 2 — TensorFlow tensors": ROOT / "experiments" / "exp02_tensorflow.py",
        "Experiment 3 — FLAN-T5 QA": ROOT / "experiments" / "exp03_qa.py",
        "Experiment 4 — SDXL image generation": ROOT / "experiments" / "exp04_image.py",
        "Experiment 5 — MNIST VAE": ROOT / "experiments" / "exp05_vae.py",
        "Experiment 6 — DCGAN": ROOT / "experiments" / "exp06_dcgan.py",
        "Experiment 7 — StyleGAN-inspired": ROOT / "experiments" / "exp07_stylegan.py",
        "Experiment 8 — MusicGen": ROOT / "experiments" / "exp08_music.py",
        "Experiment 9 — RAG + LoRA": ROOT / "experiments" / "exp09_rag_lora.py",
        "Experiment 10 — LTX-Video": ROOT / "experiments" / "exp10_video.py",
        "Unified Streamlit UI": ROOT / "app.py",
    }
    source_choice = st.selectbox("Select source file", list(source_map), key="source_browser_select")
    source_path = source_map[source_choice]
    if source_path.exists():
        source_text = source_path.read_text(encoding="utf-8")
        st.code(source_text, language="python")
        st.download_button(
            "Download selected source file",
            data=source_text.encode("utf-8"),
            file_name=source_path.name,
            mime="text/x-python",
            key="source_browser_download",
        )
    else:
        st.warning("The selected source file was not found in this installation.")
