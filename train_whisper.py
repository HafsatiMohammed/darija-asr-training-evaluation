import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import librosa
import evaluate

from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)


print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current cuda device:", torch.cuda.current_device())
    print("gpu name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("running on CPU")


# CONFIG


MODEL_NAME = "openai/whisper-small" # or  "openai/whisper-base" or any model if you have the computational power for it
DATASET_PATH = "./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only"
OUTPUT_DIR = "./whisper-small-darija" # or "./whisper-base-darija"

LANGUAGE = "arabic" # the colosest language to darija is regular arabic
# Note that data should be writen in arabic and not latin as many Moroccan datasets are 
TASK = "transcribe"
TARGET_SR = 16000

AUDIO_PATH_COL = "audio_path_16k"
TEXT_COL = "text_normalized"  # if text already normalized which is the case with my data

MAX_LABEL_LENGTH = 225
SEED = 42

# training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 5000*2
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 50

set_seed(SEED)



# SIMPLE TEXT NORMALIZATION

def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text



# LOAD DATASET

dataset = load_from_disk(DATASET_PATH)

print(dataset)
for split in dataset.keys():
    print(split, len(dataset[split]))

for split in dataset.keys():
    missing = [c for c in [AUDIO_PATH_COL, TEXT_COL] if c not in dataset[split].column_names]
    if missing:
        raise ValueError(f"Split '{split}' is missing columns: {missing}")

# Keep only the columns needed before map
for split in dataset.keys():
    dataset[split] = dataset[split].select_columns([AUDIO_PATH_COL, TEXT_COL])



# PROCESSOR

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK,
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK,
)


# PREPARE FEATURES

def prepare_example(batch: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = batch[AUDIO_PATH_COL]
    text = normalize_text(batch[TEXT_COL])

    wav, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    #print(wav.shape)
    wav = np.asarray(wav, dtype=np.float32)

    batch["input_features"] = feature_extractor(
        wav,
        sampling_rate=TARGET_SR,
    ).input_features[0]

    batch["labels"] = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
    ).input_ids

    return batch


for split in dataset.keys():
    dataset[split] = dataset[split].map(
        prepare_example,
        remove_columns=dataset[split].column_names,
        num_proc=1,
        desc=f"Preparing {split}",
    )



# DATA COLLATOR

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch



# METRICS FOR AVALUATION

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalize_text(x) for x in pred_str]
    label_str = [normalize_text(x) for x in label_str]

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "cer": cer,
    }



# MODEL

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.config.use_cache = False


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("model device:", next(model.parameters()).device)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)



# TRAINING ARGS

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    predict_with_generate=True,
    generation_max_length=MAX_LABEL_LENGTH,
    save_total_limit=3,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to=["tensorboard"],
    push_to_hub=False,
)


# TRAINER

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)


# TRAIN

train_result = trainer.train()

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "train_result.json"), "w", encoding="utf-8") as f:
    json.dump(train_result.metrics, f, ensure_ascii=False, indent=2)



# FINAL EVAL

val_metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="val")
test_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")

print("\nValidation metrics:")
print(json.dumps(val_metrics, indent=2, ensure_ascii=False))

print("\nTest metrics:")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "validation": val_metrics,
            "test": test_metrics,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nDone.")
print(f"Saved model to: {OUTPUT_DIR}")
