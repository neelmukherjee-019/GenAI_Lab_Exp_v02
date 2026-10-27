"""Experiment 9: Personalized AI assistant using RAG and optional LoRA fine-tuning.

The RAG path uses sentence-transformer embeddings and cosine similarity over chunks.
The optional training path is a small educational LoRA fine-tune; it does not replace
retrieval, and its results depend on the quality/size of the supplied examples.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import json

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QA_MODEL = "google/flan-t5-base"
DEFAULT_LORA_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def extract_document_text(file_path: str | Path) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if ext in {".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError("Supported document types: TXT, PDF, DOCX, MD.")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks with whitespace-aware boundaries."""
    text = " ".join((text or "").split())
    if not text:
        return []
    chunk_size = max(150, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_at = text.rfind(" ", start + chunk_size // 2, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def retrieve_relevant_chunks(question: str, chunks: list[str], embedder, top_k: int = 3) -> list[dict[str, Any]]:
    import numpy as np
    if not chunks:
        raise ValueError("No document chunks are available. Upload a non-empty document first.")
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    query_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
    scores = np.asarray(embeddings) @ np.asarray(query_vec)
    indices = np.argsort(scores)[::-1][:max(1, int(top_k))]
    return [{"chunk": chunks[int(i)], "score": float(scores[int(i)]), "index": int(i)} for i in indices]


def answer_base_and_rag(question: str, chunks: list[str], embedder, qa_pipeline, top_k: int = 3, max_new_tokens: int = 180) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Enter a question first.")
    question = question.strip()
    base_prompt = f"Answer the question clearly. If you do not know, say so.\nQuestion: {question}"
    base = qa_pipeline(base_prompt, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=4)[0]["generated_text"].strip()
    retrieved = retrieve_relevant_chunks(question, chunks, embedder, top_k=top_k)
    context = "\n\n".join(f"[Document excerpt {i+1}] {item['chunk']}" for i, item in enumerate(retrieved))
    rag_prompt = (
        "Answer the question using only the context below. If the answer is not in the context, "
        "say that the document does not provide enough information. Do not invent facts.\n\n"
        f"Context:\n{context[:2200]}\n\nQuestion: {question}\nAnswer:"
    )
    rag = qa_pipeline(rag_prompt, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=4)[0]["generated_text"].strip()
    return {"base_answer": base, "rag_answer": rag, "retrieved": retrieved}


def parse_qa_examples(text: str) -> list[dict[str, str]]:
    """Parse tab-separated question/answer lines; skips blank/comment lines."""
    examples = []
    for line in (text or "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError("Each non-empty training line must be: question<TAB>answer")
        examples.append({"question": parts[0].strip(), "answer": parts[1].strip()})
    if not examples:
        raise ValueError("Provide at least one question<TAB>answer training pair.")
    return examples


def fine_tune_lora(examples: list[dict[str, str]], output_dir: str | Path = "outputs/exp09/lora_adapter", base_model: str = DEFAULT_LORA_MODEL, epochs: int = 1, max_length: int = 256, learning_rate: float = 2e-4, use_4bit: bool = False) -> dict[str, Any]:
    """Fine-tune a small causal LLM with LoRA using a manual PyTorch loop.

    For QLoRA, enable `use_4bit=True` only with a CUDA GPU and bitsandbytes installed.
    The base model is downloaded from the Hugging Face Hub on first run.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    if not examples:
        raise ValueError("Training examples are required.")
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("QLoRA 4-bit mode requires a CUDA GPU in this configuration.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    device = next(model.parameters()).device

    class QADataset(Dataset):
        def __init__(self, rows):
            self.rows = []
            for row in rows:
                instruction = f"### Instruction:\n{row['question']}\n### Response:\n"
                answer = row["answer"].strip() + (tokenizer.eos_token or "")
                prompt_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
                encoded = tokenizer(instruction + answer, truncation=True, max_length=int(max_length), padding="max_length")
                labels = list(encoded["input_ids"])
                mask_len = min(len(prompt_ids), len(labels))
                labels[:mask_len] = [-100] * mask_len
                # Do not include padded positions in the loss.
                labels = [token if mask else -100 for token, mask in zip(labels, encoded["attention_mask"])]
                # Keep at least one supervised token if truncation consumed the response.
                if all(x == -100 for x in labels):
                    last_real = max(0, sum(encoded["attention_mask"]) - 1)
                    labels[last_real] = encoded["input_ids"][last_real]
                self.rows.append({
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]

    dataset = QADataset(examples)
    batch_size = min(2, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(learning_rate))
    epoch_losses = []
    model.train()
    for epoch in range(int(epochs)):
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(losses) / max(1, len(losses)))

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training_summary.json").write_text(json.dumps({"base_model": base_model, "examples": len(examples), "epochs": int(epochs), "epoch_losses": epoch_losses, "use_4bit": use_4bit}, indent=2), encoding="utf-8")
    return {"adapter_dir": str(out), "base_model": base_model, "epoch_losses": epoch_losses, "training_examples": len(examples)}


def answer_with_lora(question: str, adapter_dir: str | Path, max_new_tokens: int = 160) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    adapter_path = Path(adapter_dir)
    summary_path = adapter_path / "training_summary.json"
    base_model = DEFAULT_LORA_MODEL
    if summary_path.exists():
        base_model = json.loads(summary_path.read_text(encoding="utf-8")).get("base_model", base_model)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    base.to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    prompt = f"### Instruction:\n{question.strip()}\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text


if __name__ == "__main__":
    print("Experiment 9 module loaded. Use extract_document_text(), chunk_text(), RAG functions, and optional fine_tune_lora().")
