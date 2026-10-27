
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"


class Generator:
    def __init__(self):
        print(f"Loading generation model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, question, context, instruction=None):
        instruction_text = ""
        if instruction:
            instruction_text = (
                f"\nAdditional instruction:\n{instruction}\n"
            )

        prompt = f"""
Answer the question using only the provided context.
{instruction_text}
Context:
{context}

Question:
{question}

Answer:
"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
