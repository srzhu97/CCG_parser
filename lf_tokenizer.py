from nltk.sem.logic import LogicParser, Tokens
import numpy as np
import sys

class LFTokenizer:
    START = "<start>"
    END = "<end>"
    UNK = "<unk>"
    NULL = "<null>"

    CUSTOM_KEYWORDS = [
        "True",
        START,
        END,
        UNK,
        NULL
        # anything else?
    ]

    VARIABLES = [
        "x", "d", "F"
        # anything else?
    ]

    def __init__(self):
        #   Map keyword indices to keywords.
        self.idx2kw = [
            Tokens.LAMBDA,
            Tokens.EXISTS,
            Tokens.ALL,
            Tokens.DOT,
            Tokens.OPEN,
            Tokens.CLOSE,
            Tokens.COMMA,
            Tokens.NOT,
            Tokens.AND,
            Tokens.OR,
            Tokens.IMP,
            Tokens.IFF,
            Tokens.EQ,
            Tokens.NEQ,
        ] + LFTokenizer.CUSTOM_KEYWORDS + LFTokenizer.VARIABLES

        #   List of sets of keywords; keywords in each set should map to the
        #   same index.
        kwsets = [
            Tokens.LAMBDA_LIST,
            Tokens.EXISTS_LIST,
            Tokens.ALL_LIST,
            [ Tokens.DOT ],
            [ Tokens.OPEN ],
            [ Tokens.CLOSE ],
            [ Tokens.COMMA ],
            Tokens.NOT_LIST,
            Tokens.AND_LIST,
            Tokens.OR_LIST,
            Tokens.IMP_LIST,
            Tokens.IFF_LIST,
            Tokens.EQ_LIST,
            Tokens.NEQ_LIST,
        ] + [ [ x ] for x in LFTokenizer.CUSTOM_KEYWORDS ] + [ [ x ] for x in LFTokenizer.VARIABLES ]

        #   Keyword space, i.e. set of all keywords.
        self.kw_space = set()

        #   Map keywords to keyword indices.
        self.kw2idx = dict()
        for k, kwset in enumerate(kwsets):
            for kw in kwset:
                self.kw_space.add(kw)
                self.kw2idx[kw] = k

        #   Symbol info... to be constructed.
        self.minfreq   = None
        self.sym_space = None
        self.idx2sym   = None
        self.sym2idx   = None

    def get_variable_info(tok):
        """
        Return the information of a token which is potentially a variable.

        :return: `None` if token is not a variable. If the token is a variable,
            return a tuple `(variable, idx)` where `variable` is the string
            that represents the type of variable and `idx` is the integer index
            of the variable. For example: `get_variable_info("x5")` gives you
            `("x", 5)`.

        """

        for variable in LFTokenizer.VARIABLES:
            if not(tok.startswith(variable)):
                continue

            try:
                idx = int(tok[len(variable):])
            except:
                continue

            if idx >= 0:
                return (variable, idx)

        return None

    def build_syminfo(self, exprs, minfreq=1):
        """
        Build symbol information from a list of expressions.

        :param exprs: List of expressions (strings).
        :param minfreq: A symbol has to appear at least `minfreq`-many times in
            the corpus in order to be added to the vocabulary.
        """

        self.minfreq = minfreq

        #   Mapping from symbols to their frequencies.
        sym2freq = dict()

        #   Logic parser from nltk.sem.logic.
        parser = LogicParser()

        for expr in exprs:
            #   Parse expression into tokens.
            tokens = parser.process(expr)[0]
            
            for tok in tokens:
                if not(tok in self.kw_space) and (LFTokenizer.get_variable_info(tok) is None):
                    tok = tok.lower()

                    if not(tok in sym2freq):
                        sym2freq[tok] = 0

                    sym2freq[tok] += 1

        #   Symbol space, i.e. set of all symbols.
        self.sym_space = set()

        #   Only add symbols that have appeared at least `minfreq`-many times.
        for sym, freq in sym2freq.items():
            if freq >= minfreq:
                self.sym_space.add(sym)

        #   Number of unique keyword indices. This tells us how much to offset
        #   the symbol indices by.
        num_unique_kwidxs = len(set(self.kw2idx.values()))

        #   Map symbol indices to symbols. The first elements of this map is a
        #   list of `None`s because we need to offset the symbol indices by the
        #   number of unique keyword indices.
        self.idx2sym = [ None for _ in range(num_unique_kwidxs) ] + list(self.sym_space)

        #   Map symbols to symbol indices.
        self.sym2idx = dict()
        for k, sym in enumerate(self.idx2sym):
            if sym is None:
                continue

            self.sym2idx[sym] = k

    def dump_syminfo(self):
        """
        Return a dictionary containing information about this tokenizer. Use
        `load_syminfo` to load the returned dictionary into an initialized
        tokenizer.
        """

        return {
            "minfreq":   self.minfreq,
            "sym_space": self.sym_space,
            "idx2sym":   self.idx2sym,
            "sym2idx":   self.sym2idx
        }

    def load_syminfo(self, syminfo):
        """
        Load information.
        """

        self.minfreq   = syminfo["minfreq"]
        self.sym_space = syminfo["sym_space"]
        self.idx2sym   = syminfo["idx2sym"]
        self.sym2idx   = syminfo["sym2idx"]

    def tokenize(self, exprs, wrap_start_end=False, expository=False, non_var_idx=0, pad_token_idx=0, pad_var_idx=0):
        #   Lists of sequences.
        expository_seqs = []
        token_idx_seqs = []
        var_idx_seqs = []
        token_var_idx_lens = []

        #   Maximum sequence length, useful for padding later.
        maxlen = 0

        #   Token parser from nltk.sem.logic.
        parser = LogicParser()

        for expr in exprs:
            #   If LF is null, pass all of this.
            if expr.strip() == LFTokenizer.NULL:
                expository_seqs.append(["Null"])
                token_idx_seqs.append(np.array([self.kw2idx[LFTokenizer.NULL]], dtype=np.int64))
                var_idx_seqs.append(np.array([non_var_idx], dtype=np.int64))
                token_var_idx_lens.append(1)
                maxlen = max(maxlen, 1)

                continue

            #   Parse expression into a list of tokens.
            tokens = parser.process(expr)[0]

            #   Expository sequence: list of human-friendly strings that tell
            #   you what the tokenizer considers what type each token is.
            expository_seq = []

            #   Token index sequence.
            token_idx_seq = []

            #   Variable index sequence. If a token is not a variable, the
            #   variable index is `non_var_idx`.
            var_idx_seq = []

            #   Start with a start token if `wrap_start_end`.
            if wrap_start_end:
                expository_seq.append(f"Start")
                token_idx_seq.append(self.kw2idx[LFTokenizer.START])
                var_idx_seq.append(non_var_idx)

            for tok in tokens:
                #   If the token is a keyword...
                if tok in self.kw_space:
                    expository_seq.append(f"Kw(\"{tok}\")")
                    token_idx_seq.append(self.kw2idx[tok])
                    var_idx_seq.append(non_var_idx)
                    continue

                #   Or, if the token is a variable...
                var_idx_pair = LFTokenizer.get_variable_info(tok)
                if var_idx_pair is not None:
                    var, var_idx = var_idx_pair
                    expository_seq.append(f"Var(\"{var}\")")
                    token_idx_seq.append(self.kw2idx[var])
                    var_idx_seq.append(var_idx)
                    continue

                tok = tok.lower()

                #   Or, if the token is a symbol...
                if tok in self.sym_space:
                    expository_seq.append(f"Sym(\"{tok}\")")
                    token_idx_seq.append(self.sym2idx[tok])
                    var_idx_seq.append(non_var_idx)
                    continue

                #   If all else fails, the token is an unknown token.
                expository_seq.append(f"Unk(\"{tok}\")")
                token_idx_seq.append(self.kw2idx[LFTokenizer.UNK])
                var_idx_seq.append(non_var_idx)

            #   End with an end token if `wrap_start_end`.
            if wrap_start_end:
                expository_seq.append(f"End")
                token_idx_seq.append(self.kw2idx[LFTokenizer.END])
                var_idx_seq.append(non_var_idx)

            #   Push sequences and updated maxlen.
            expository_seqs.append(expository_seq)
            token_idx_seqs.append(np.array(token_idx_seq, dtype=np.int64))
            var_idx_seqs.append(np.array(var_idx_seq, dtype=np.int64))
            token_var_idx_lens.append(len(token_idx_seq))
            maxlen = max(maxlen, len(token_idx_seq))

        #   Pad the expository sequences.
        padded_expository_seqs = []
        for expository_seq in expository_seqs:
            padded_expository_seqs.append(
                expository_seq + [ "Pad" for _ in range(maxlen - len(expository_seq)) ]
            )

        #   Pad the token index sequences.
        padded_token_idx_seqs = []
        for token_idx_seq in token_idx_seqs:
            padded_token_idx_seqs.append(
                np.pad(token_idx_seq, (0, maxlen - len(token_idx_seq)), "constant", constant_values=(pad_token_idx, pad_token_idx))
            )

        #   Pad the variable index sequences.
        padded_var_idx_seqs = []
        for var_idx_seq in var_idx_seqs:
            padded_var_idx_seqs.append(
                np.pad(var_idx_seq, (0, maxlen - len(var_idx_seq)), "constant", constant_values=(pad_var_idx, pad_var_idx))
            )

        token_var_idx_lens = np.array(token_var_idx_lens, dtype=np.int64)

        #   Stack the token/variable index sequences.
        stacked_token_idx_seqs = np.stack(padded_token_idx_seqs)
        stacked_var_idx_seqs = np.stack(padded_var_idx_seqs)
        assert stacked_token_idx_seqs.shape == stacked_var_idx_seqs.shape

        if expository:
            return stacked_token_idx_seqs, stacked_var_idx_seqs, token_var_idx_lens, padded_expository_seqs
        else:
            return stacked_token_idx_seqs, stacked_var_idx_seqs, token_var_idx_lens

    def check(self):
        kw2idx_keys = set(self.kw2idx.keys())
        for kw in Tokens.TOKENS:
            assert kw in self.kw_space, f"self.kw_space does not contain the keyword \"{kw}\". This keyword is in Tokens.TOKENS."
            assert kw in kw2idx_keys, f"self.kw2idx does not contain the keyword \"{kw}\" in its keys. This keyword is in Tokens.TOKENS."

        for kw in Tokens.SYMBOLS:
            assert kw in self.kw_space, f"self.kw_space does not contain the keyword \"{kw}\". This keyword is in Tokens.SYMBOLS."
            assert kw in kw2idx_keys, f"self.kw2idx does not contain the keyword \"{kw}\" in its keys. This keyword is in Tokens.SYMBOLS."

        for kw in LFTokenizer.CUSTOM_KEYWORDS:
            assert kw in self.kw_space, f"self.kw_space does not contain the keyword \"{kw}\". This keyword is in LFVocab.CUSTOM_KEYWORDS."
            assert kw in kw2idx_keys, f"self.kw2idx does not contain the keyword \"{kw}\" in its keys. This keyword is in LFTokenizer.CUSTOM_KEYWORDS."

        for kw in LFTokenizer.VARIABLES:
            assert kw in self.kw_space, f"self.kw_space does not contain the keyword \"{kw}\". This keyword is in LFVocab.VARIABLES."
            assert kw in kw2idx_keys, f"self.kw2idx does not contain the keyword \"{kw}\" in its keys. This keyword is in LFTokenizer.VARIABLES."

        if not(self.sym_space is None):
            sym2idx_keys = set(self.sym2idx.keys())

            # sym_space \subsetof idx2sym, sym2idx.keys()?
            for sym in self.sym_space:
                assert sym in self.idx2sym, f"Symbol \"{sym}\" in self.sym_space but not in self.idx2sym."
                assert sym in sym2idx_keys, f"Symbol \"{sym}\" in self.sym_space but not in self.sym2idx as a key."

            # idx2sym \subsetof sym_space, sym2idx.keys()?
            for sym in self.idx2sym:
                if sym is None:
                    continue

                assert sym in self.sym_space, f"Symbol \"{sym}\" in self.idx2sym but not in self.sym_space."
                assert sym in sym2idx_keys, f"Symbol \"{sym}\" in self.idx2sym but not in self.sym2idx as a key."

            # sym2idx.keys() \subsetof sym_space, idx2sym?
            for sym in sym2idx_keys:
                assert sym in self.sym_space, f"Symbol \"{sym}\" in self.sym2idx as a key but not in self.sym_space."
                assert sym in self.idx2sym, f"Symbol \"{sym}\" in self.sym2idx as a key but not in self.idx2sym."

    def summary(self):
        res = ""

        num_kws = len(self.kw_space)
        num_unique_kwidxs = len(set(self.kw2idx.values()))
        sorted_kw2idx_items = sorted(
            self.kw2idx.items(),
            key=lambda x: x[1]
        )

        res += f"{num_kws} keywords, {num_unique_kwidxs} unique indices:\n"
        for kw, k in sorted_kw2idx_items:
            res += f"    Keyword \"{kw}\" maps to index {k}\n"

        num_syms = len(self.sym_space)
        num_unique_symidxs = len(set(self.sym2idx.values()))
        sorted_sym2idx_items = sorted(
            self.sym2idx.items(),
            key=lambda x: x[1]
        )

        res += f"{num_syms} symbols, {num_unique_symidxs} unique indices:\n"
        for sym, k in sorted_sym2idx_items:
            res += f"    Symbol \"{sym}\" maps to index {k}\n"

        return res


# testing code
if __name__ == "__main__":
    import json
    from prettytable import PrettyTable

    import torch

    exprs = []
    exprs_json_path = sys.argv[1]
    with open(exprs_json_path, "r") as f:
        for expr_json_raw in f:
            expr_json = json.loads(expr_json_raw)
            exprs.append(expr_json["lf1"].strip())
            exprs.append(expr_json["lf2"].strip())

    lftoker = LFTokenizer()
    lftoker.check()

    lftoker.build_syminfo(exprs)
    lftoker.check()
    print(lftoker.summary())

    syminfo = lftoker.dump_syminfo()
    torch.save(syminfo, "syminfo.pt")

    exit()

    tokenized = lftoker.tokenize(exprs, wrap_start_end=True, expository=True, non_var_idx=0)

    for expr, token_idx_seq, var_idx_seq, annotation_seq in zip(exprs, tokenized[0], tokenized[1], tokenized[2]):
        print("=" * 80)

        print("Expression:")
        print(expr)

        annotation_seq += [ "" * (len(token_idx_seq) - len(annotation_seq)) ]

        ptable = PrettyTable()
        ptable.field_names = [ "Token index", "Var index", "Token" ]
        for token_idx, var_idx, annotation in zip(token_idx_seq, var_idx_seq, annotation_seq):
            ptable.add_row([ token_idx, var_idx, annotation ])

        print(ptable)
        print("")