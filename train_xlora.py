"""
Train the xLoRA classifier on the LoRAuter validation set.

Only the xLoRA classifier is trained; the base model and all LoRA adapters
remain frozen throughout. The trained checkpoint is saved to OUTPUT_DIR and
used by run_inference.py.
"""

import json
import torch
import xlora
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
TRAIN_DATA = "dataset/config_large_flat.json"
OUTPUT_DIR = "xlora_trained"
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4          # effective batch = 16
EPOCHS = 3
LR = 1e-4

# All 48 NLP task adapters (igzi/lora-<task> on HuggingFace, base: Llama-3.1-8B)
TASK_ADAPTERS = {
    "anli_r1":                    "igzi/lora-anli_r1",
    "anli_r2":                    "igzi/lora-anli_r2",
    "anli_r3":                    "igzi/lora-anli_r3",
    "arc_challenge":              "igzi/lora-arc_challenge",
    "arc_easy":                   "igzi/lora-arc_easy",
    "bool_q":                     "igzi/lora-bool_q",
    "cb":                         "igzi/lora-cb",
    "common_gen":                 "igzi/lora-common_gen",
    "copa":                       "igzi/lora-copa",
    "cosmos_qa":                  "igzi/lora-cosmos_qa",
    "dart":                       "igzi/lora-dart",
    "definite_pronoun_resolution":"igzi/lora-definite_pronoun_resolution",
    "drop":                       "igzi/lora-drop",
    "e2e_nlg":                    "igzi/lora-e2e_nlg",
    "glue_mrpc":                  "igzi/lora-glue_mrpc",
    "glue_qqp":                   "igzi/lora-glue_qqp",
    "hellaswag":                  "igzi/lora-hellaswag",
    "imdb_reviews":               "igzi/lora-imdb_reviews",
    "mnli_matched":               "igzi/lora-mnli_matched",
    "mnli_mismatched":            "igzi/lora-mnli_mismatched",
    "multirc":                    "igzi/lora-multirc",
    "natural_questions":          "igzi/lora-natural_questions",
    "openbookqa":                 "igzi/lora-openbookqa",
    "para_crawl_enes":            "igzi/lora-para_crawl_enes",
    "paws_wiki":                  "igzi/lora-paws_wiki",
    "piqa":                       "igzi/lora-piqa",
    "qnli":                       "igzi/lora-qnli",
    "record":                     "igzi/lora-record",
    "rte":                        "igzi/lora-rte",
    "sentiment140":               "igzi/lora-sentiment140",
    "snli":                       "igzi/lora-snli",
    "squad_v1":                   "igzi/lora-squad_v1",
    "squad_v2":                   "igzi/lora-squad_v2",
    "sst2":                       "igzi/lora-sst2",
    "stsb":                       "igzi/lora-stsb",
    "story_cloze":                "igzi/lora-story_cloze",
    "trivia_qa":                  "igzi/lora-trivia_qa",
    "web_nlg_en":                 "igzi/lora-web_nlg_en",
    "wnli":                       "igzi/lora-wnli",
    "wmt14_enfr":                 "igzi/lora-wmt14_enfr",
    "wmt16_translate_deen":       "igzi/lora-wmt16_translate_deen",
    "wmt16_translate_fien":       "igzi/lora-wmt16_translate_fien",
    "wmt16_translate_roen":       "igzi/lora-wmt16_translate_roen",
    "wmt16_translate_ruen":       "igzi/lora-wmt16_translate_ruen",
    "wmt16_translate_tren":       "igzi/lora-wmt16_translate_tren",
    "wsc":                        "igzi/lora-wsc",
    "yelp_polarity_reviews":      "igzi/lora-yelp_polarity_reviews",
}

# Alpaca prompt template (matches adapter fine-tuning format)
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_example(example: dict) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=example["inputs"])
    return prompt + example["targets"]


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )

    model = xlora.add_xlora_to_model(
        model=model,
        xlora_config=xlora.xLoRAConfig(
            hidden_size=config.hidden_size,
            base_model_id=BASE_MODEL,
            xlora_depth=8,
            device=torch.device("cuda"),
            adapters=TASK_ADAPTERS,
        ),
        verbose=True,
    )
    model.print_trainable_parameters()

    with open(TRAIN_DATA) as f:
        raw = json.load(f)

    texts = [format_example(ex) for ex in raw]
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"]})

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
