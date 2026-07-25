import re

SAPROT_MODEL_TYPES = {"saprot", "saprot_mlp"}
ESM2_MODEL_TYPES = {"esm2", "esm2_mlp"}
ESMC_MODEL_TYPES = {"esmc", "esmc_mlp"}
SEQDANCE_MODEL_TYPES = {"seqdance", "esmdance"}
MLP_MODEL_TYPES = {"esm2_mlp", "saprot_mlp", "esmc_mlp"}

FUSION_MODEL_TYPES = ("saprot", "esm2", "esmc", "seqdance", "esmdance")
TRAIN_INFERENCE_MODEL_TYPES = FUSION_MODEL_TYPES + ("esm2_mlp", "saprot_mlp", "esmc_mlp")
ANNOTATION_MODEL_TYPES = FUSION_MODEL_TYPES

HF_SEQUENCE_MODEL_TYPES = {"saprot", "esm2", "seqdance", "esmdance"}
NON_SAPROT_HF_MODEL_TYPES = HF_SEQUENCE_MODEL_TYPES - {"saprot"}
SEQUENCE_INPUT_MODEL_TYPES = {"esm2", "esmc", "seqdance", "esmdance", "esm2_mlp", "esmc_mlp"}

SEQDANCE_HIDDEN_SIZE = 480
SEQDANCE_DEFAULT_CHECKPOINTS = {
    "seqdance": "ChaoHou/SeqDance",
    "esmdance": "ChaoHou/ESMDance",
}

MODEL_TYPE_INFERENCE_ORDER = ("esmdance", "seqdance", "saprot", "esmc", "esm2")


def infer_model_type_from_checkpoint(checkpoint_path):
    tokens = set(re.split(r"[\\/_.-]+", checkpoint_path.lower()))
    for model_type in MODEL_TYPE_INFERENCE_ORDER:
        if model_type in tokens:
            return model_type
    return None
