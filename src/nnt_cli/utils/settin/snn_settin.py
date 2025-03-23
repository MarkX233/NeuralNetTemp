
def decode_reset_mode(reset_mode):
        mapping = {0: "zero", 1: "subtract"}
        return tuple(mapping[val] for val in reset_mode)