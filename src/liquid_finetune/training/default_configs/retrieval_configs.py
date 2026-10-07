from liquid_finetune import BASE_OUTPUT_PATH


_DEFAULT_RETRIEVAL = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": True,
    "gradient_accumulation_steps": 1,
    "gather_across_devices": True,
}

DEFAULT_EMBEDDING = {
    **_DEFAULT_RETRIEVAL,
    "training_type": "embedding",
    "output_dir": BASE_OUTPUT_PATH / "embedding",
    "loss": "multiple_negatives_ranking",
    "prompts": {
        "query": "query: ",
        "positive": "document: ",
        "negative": "document: ",
    },
}

DEFAULT_COLBERT = {
    **_DEFAULT_RETRIEVAL,
    "training_type": "colbert",
    "output_dir": BASE_OUTPUT_PATH / "colbert",
    "loss": "contrastive",
    "temperature": 0.02,
}
