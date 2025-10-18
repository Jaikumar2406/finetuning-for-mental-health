# Fine-Tuned TinyLlama for Mental Health Counseling 💚

This repository contains a fine-tuned version of **TinyLlama-1.1B-Chat-v1.0**, optimized for **mental health and counseling conversations**. The model is trained to provide empathetic and supportive responses in a safe and responsible manner.

---

## 📚 Dataset

The fine-tuning dataset is named:

**`mental_health_counseling_conversations`**  

It consists of anonymized and safe counseling dialogues intended for mental health support. This dataset is used only for research and educational purposes.

---

## 🧠 Model Details

- **Base Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0  
- **Fine-Tuning Technique:** LoRA / PEFT  
- **Training Goal:** Mental health counseling and empathetic response generation  
- **Adapter Weights Directory:** `results_adapter/`

---

## ⚡ Installation

```bash
git clone https://github.com/Jaikumar2406/finetuning-for-mental-health.git
cd finetuning-for-mental-health

# Recommended: create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## 🏋️ Training
**If you want to fine-tune yourself or continue training:**
python train.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name mental_health_counseling_conversations \
    --output_dir results_adapter \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2

    
## 🤖 Inference ##
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "results_adapter"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=250,
)

prompt = "I am feeling very anxious today. Can you help me?"
response = pipe(f"[INST] {prompt} [/INST]")

print(response[0]['generated_text'])


## ⚠️ Safety & Disclaimer
- This model is not a substitute for professional mental health advice.
- Use this model responsibly and ethically.
- For any severe mental health issues, consult a licensed therapist or professional.

<img width="1855" height="735" alt="Screenshot 2025-10-19 011757" src="https://github.com/user-attachments/assets/fc57706d-0446-42a9-a33f-016068ef0f99" />

