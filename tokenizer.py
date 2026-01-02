import re

class CustomTokenizer:
    def __init__(self):
        self.token_to_id = {"[PAD]": 0, "[SEP]": 1}
        self.id_to_token = {0: "[PAD]", 1: "[SEP]"}
        self.vocab_size = 2

    def tokenize(self, code_line):
        # Tokenize identifiers, numbers, and common punctuation/operators safely
        pattern = r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|==|!=|<=|>=|[=+\-\*/<>()[\]{},.:;]|\"[^\"]*\"|'[^']*'"
        return re.findall(pattern, code_line)

    def build_vocab(self, code_lines):
        for line in code_lines:
            for token in self.tokenize(line):
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, code_line, max_length=512):
        tokens = self.tokenize(code_line)
        ids = [self.token_to_id.get(token, 0) for token in tokens] + [self.token_to_id["[SEP]"]]
        return ids[:max_length]  # truncate long sequences

    def decode(self, token_ids):
        return [self.id_to_token.get(i, "[UNK]") for i in token_ids if i in self.id_to_token]