import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


class Generator:
    def __init__(self):
        print(f"Loading generation model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        # Explicitly use CPU.
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.model.eval()

    def generate(self, question, context):

        prompt = f"""
You are a question-answering system for a retrieval-augmented
generation system.

Answer the question using ONLY the information explicitly stated
in the context.

Do NOT:
- use outside knowledge
- infer information that is not explicitly stated
- confuse "where" with "when"
- confuse "who" with "what"
- guess an answer

If the context does not explicitly contain the answer, respond:
"The answer is not specified in the provided context."

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

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer