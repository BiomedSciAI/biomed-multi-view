import ast


def fix_finetuning_args(finetuning_args):
    for k, v in finetuning_args.items():
        try:
            value = ast.literal_eval(v)
        except Exception:
            value = v
        finetuning_args[k] = value
    return finetuning_args
