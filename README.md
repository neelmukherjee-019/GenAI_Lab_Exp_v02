GenAI Lab Suite — Experiments 1–10
This project contains ten Python experiment modules and one Streamlit UI that wraps them into a single interactive workbench. The UI includes a source-code browser so you can view and download the experiment modules from within the app. It is based on the lab manuals supplied in the conversation.
Included experiments
Dataset analysis and preprocessing for CSV/Excel files, with an optional local Hugging Face + LangChain narrative summary.
Basic Python operations and TensorFlow 2 tensor operations. The uploaded PDF filename is Experiment No. 2, although the title inside it says Experiment No. 1; this project labels it as Experiment 2 to keep the files sequential.
FLAN-T5 pretrained question-answer system.
Stable Diffusion XL text-to-image generation.
TensorFlow/Keras VAE trained on MNIST, with reconstructions, 2D latent-space plot, generated digits and saved encoder/decoder.
PyTorch DCGAN trained on a supplied folder/ZIP of face images.
Educational StyleGAN-inspired model demonstrating mapping, layer-wise styles, style mixing and noise injection. It is not NVIDIA's full StyleGAN2-ADA implementation.
Pretrained MusicGen text-to-music generation.
Document-based RAG assistant (base vs RAG answers) and optional LoRA/QLoRA fine-tuning.
LTX-Video text-to-video generation.
Run the UI
```bash
cd GenAI_Lab_Suite
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```
Install a CUDA-compatible PyTorch build appropriate for your machine from the official PyTorch selector if you want GPU support. Google Colab GPU is recommended for the larger generative experiments. Do not expect heavy diffusion/video/audio models to run quickly on CPU.
Dataset uploads
Experiment 1 accepts `.csv`, `.xlsx`, or `.xls`.
Experiments 6 and 7 accept a ZIP containing JPG/PNG/WebP face images. A CelebA archive may be used; the ZIP can contain nested directories.
Experiment 9 accepts `.txt`, `.pdf`, `.docx`, `.md`, or `.markdown`. Scanned PDFs without embedded text may require OCR and will not produce useful chunks in this basic implementation.
Standalone module examples
The modules are designed so they can be imported from Python as well as called by the UI. Run from the project root:
```python
from experiments.exp01_dataset import analyze_dataset
result = analyze_dataset("my_data.csv")

from experiments.exp02_tensorflow import run_tensorflow_demo
result = run_tensorflow_demo()

from experiments.exp03_qa import load_qa_pipeline, answer_question
qa = load_qa_pipeline()
print(answer_question("What is RAG?", qa))
```
For Experiments 4, 8 and 10, load their pretrained pipeline and pass it into the corresponding generation function. For Experiments 5–7, call the `train_*` function with the requested settings and dataset location.
Notes and limitations
First use of pretrained Hugging Face models downloads model weights; an internet connection is required.
SDXL and LTX-Video are large models. The LTX-Video documentation estimates about 10 GB VRAM for the documented configuration.
Experiment 5 defaults to a short three-epoch run for a quick interactive demo. The lab manual recommends 30 epochs; select 30 for the longer run.
GAN quality depends on dataset size, number of epochs, compute resources and training stability. Short demos are for demonstrating the architecture and procedure, not for promising photorealistic output.
The LoRA trainer uses a small Qwen instruct model and tiny supplied question-answer pairs for demonstration. Fine-tuning is not a substitute for RAG: retrieval provides external facts, while fine-tuning adapts the model's response behaviour.
Use generated images/audio/video responsibly and follow the license and usage terms of the selected pretrained model.
