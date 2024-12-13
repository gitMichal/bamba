def get_pretty_name(name: str):
    if name in name2tag:
        out = name2tag[name]
    else:
        out = name.capitalize()

    return out


name2tag = {
    "mmlu": "MMLU",
    "hellaswag": "Hellaswag",
    "winogrande": "Winogrande",
    "piqa": "Piqa",
    "openbookqa": "OpenbookQA",
    "arc_challenge": "ARC-C",
    "truthfulqa_mc2": "TruthfulQA",
    "gsm8k": "GSM8K",
    "bbh": "BBH",
    "musr": "MuSR",
    "mmlu_pro": "MMLU-PRO",
    "gpqa": "GPQA",
    "math_hard": "MATH Lvl 5",
    "ifeval": "IFEval",
    "toxigen": "Toxigen",
}
