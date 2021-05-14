from lf_tokenizer import LFTokenizer
from nltk.sem.logic import LogicParser, Tokens
import re

parser = LogicParser()

def preproc(expr):
    """
    Do 4 things:

    *   Map variable indices to consecutive indices.
    *   Remove "True &".
    *   Remove "& True".
    *   Replace empty LFs with "<null>".
    """

    #   Remove "True &".
    expr = re.sub("True &", "", expr)

    #   Remove "& True"
    expr = re.sub("& True", "", expr)

    #   Parse expression into tokens.
    tokens = parser.process(expr)[0]

    #   Remap variable indices.
    var_idx_remap = dict()

    for tok in tokens:
        var_idx_pair = LFTokenizer.get_variable_info(tok)
        if var_idx_pair is not None:
            _, var_idx = var_idx_pair
            if not(var_idx in var_idx_remap):
                var_idx_remap[var_idx] = len(var_idx_remap) + 1

    new_tokens = []
    for tok in tokens:
        var_idx_pair = LFTokenizer.get_variable_info(tok)
        if var_idx_pair is None:
            new_tokens.append(tok)
            continue

        var, var_idx = var_idx_pair
        var_idx = var_idx_remap[var_idx]
        new_tokens.append(f"{var}{var_idx}")

    return " ".join(new_tokens)

if __name__ == "__main__":
    import sys
    import json

    exprs_json_path = sys.argv[1]
    with open(exprs_json_path, "r") as f:
        for expr_json in f:
            expr_dict = json.loads(expr_json)

            new_expr_dict = dict()
            for k, v in expr_dict.items():
                if k not in ("lf1", "lf2"):
                    new_expr_dict[k] = v
                else:
                    if v == "":
                        new_expr_dict[k] = "<null>"
                    else:
                        new_expr_dict[k] = preproc(v)

            if not("lf1" in new_expr_dict):
                new_expr_dict["lf1"] = "<null>"

            if not("lf2" in new_expr_dict):
                new_expr_dict["lf2"] = "<null>"

            new_expr_json = json.dumps(new_expr_dict)
            print(new_expr_json)