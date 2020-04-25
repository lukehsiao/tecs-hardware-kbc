emmental_config = {
    "meta_config": {"seed": 5, "verbose": True, "log_path": "logs"},
    "data_config": {"min_data_len": 0, "max_data_len": 0},
    "model_config": {"model_path": None, "device": 0, "dataparallel": False},
    "learner_config": {
        "fp16": False,
        "n_epochs": 20,
        "train_split": ["train"],
        "valid_split": ["valid", "test"],
        "test_split": ["test"],
        "ignore_index": None,
        "optimizer_config": {
            "optimizer": "adam",
            "lr": 0.00005,
            "l2": 0.0001,
            "adam_config": {"betas": (0.9, 0.999), "eps": 1e-08, "amsgrad": False},
            "sgd_config": {"momentum": 0, "dampening": 0, "nesterov": False},
        },
        "lr_scheduler_config": {
            "lr_scheduler": "plateau",
            "lr_scheduler_step_unit": "epoch",
            "lr_scheduler_step_freq": 1,
            "warmup_steps": None,
            "warmup_unit": "batch",
            "warmup_percentage": None,
            "min_lr": 0.0,
            "reset_state": False,
            "exponential_config": {"gamma": 0.9},
            "plateau_config": {
                "metric": "Thumbnail/Thumbnail/valid/f1",
                "mode": "max",
                "factor": 0.1,
                "patience": 0,
                "threshold": 0.0000001,
                "threshold_mode": "rel",
                "cooldown": 0,
                "eps": 1e-08,
            },
            "cosine_annealing_config": {"last_epoch": -1},
        },
        "task_scheduler_config": {
            "task_scheduler": "round_robin",
            "sequential_scheduler_config": {"fillup": False},
            "round_robin_scheduler_config": {"fillup": False},
            "mixed_scheduler_config": {"fillup": False},
        },
        "global_evaluation_metric_dict": None,
    },
    "logging_config": {
        "counter_unit": "epoch",
        "evaluation_freq": 1,
        "writer_config": {"writer": "tensorboard", "verbose": True},
        "checkpointing": True,
        "checkpointer_config": {
            "checkpoint_path": None,
            "checkpoint_freq": 1,
            "checkpoint_metric": {"Thumbnail/Thumbnail/valid/f1": "max"},
            "checkpoint_task_metrics": None,
            "checkpoint_runway": 0,
            "checkpoint_all": False,
            "clear_intermediate_checkpoints": True,
            "clear_all_checkpoints": False,
        },
    },
}