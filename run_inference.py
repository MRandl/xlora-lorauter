"""
Generate predictions on the combined test set using a trained xLoRA model.

Output is a JSON file with the same entries as the test set plus a
"prediction" field, ready for metric computation.

Usage:
    python run_inference.py [--checkpoint xlora_trained] [--output xlora_predictions.json]
"""

import argparse
import json
import torch
import xlora
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
TEST_DATA = "dataset/combined_test.json"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8

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

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_model(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batched generation

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # from_pretrained takes a plain base model, loads the saved xLoRA config
    # (adapter paths, classifier architecture) and classifier weights from
    # the checkpoint directory produced by train_xlora.py.
    model = xlora.from_pretrained(checkpoint, base, "cuda")
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    results = []
    for out in outputs:
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        results.append(text.strip())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="xlora_trained")
    parser.add_argument("--test_data", default=TEST_DATA)
    parser.add_argument("--output", default="xlora_predictions.json")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)

    with open(args.test_data) as f:
        test_data = json.load(f)

    results = []
    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Generating"):
        batch = test_data[i : i + BATCH_SIZE]
        prompts = [PROMPT_TEMPLATE.format(instruction=ex["inputs"]) for ex in batch]
        predictions = generate_batch(model, tokenizer, prompts)
        for ex, pred in zip(batch, predictions):
            results.append({
                "inputs": ex["inputs"],
                "targets": ex["targets"],
                "prediction": pred,
                "task": ex.get("task", ""),
                "domain": ex.get("domain", ""),
                "metric": ex.get("metric", ""),
            })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
