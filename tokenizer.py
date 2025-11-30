
import re

class CustomTokenizer:
    def __init__(self):
        self.token_to_id = {"[PAD]": 0, "[SEP]": 1}
        self.id_to_token = {0: "[PAD]", 1: "[SEP]"}
        self.vocab_size = 2

    def tokenize(self, code_line):
        """
        Splits both LLVM IR and Python code into tokens.
        This avoids regex errors by escaping special characters properly.
        """
        pattern = r"[A-Za-z_][A-Za-z0-9_]*" \
                  r"|[0-9]+" \
                  r"|==|!=|<=|>=|[=+\-*/<>()[\]{},.:;]"
        return re.findall(pattern, code_line)

    def build_vocab(self, code_lines):
        """Builds vocabulary from a list of code lines."""
        for line in code_lines:
            for token in self.tokenize(line):
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, code_line):
        """Encodes a line of code into token IDs."""
        tokens = self.tokenize(code_line)
        return [self.token_to_id.get(token, 0) for token in tokens] + [self.token_to_id["[SEP]"]]

    def decode(self, token_ids):
        """Decodes token IDs back into code."""
        return [self.id_to_token.get(i, "[UNK]") for i in token_ids if i in self.id_to_token]


