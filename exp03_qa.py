"""Experiment 3: FLAN-T5 text generation based question-answer system."""
from __future__ import annotations

DEFAULT_MODEL_ID = "google/flan-t5-base"


def load_qa_pipeline(model_id: str = DEFAULT_MODEL_ID):
    """Load a pretrained FLAN-T5 text-to-text generation pipeline."""
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model=model_id, device=device)


def answer_question(question: str, generator, max_new_tokens: int = 160) -> str:
    if not question or not question.strip():
        raise ValueError("Enter a question first.")
    prompt = "Answer the question clearly and accurately. If uncertain, say so.\nQuestion: " + question.strip()
    result = generator(prompt, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=4)
    return result[0]["generated_text"].strip()


if __name__ == "__main__":
    model = load_qa_pipeline()
    print(answer_question("What is a transformer model?", model))
