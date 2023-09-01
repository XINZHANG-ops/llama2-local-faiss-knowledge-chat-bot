import hashlib


# this function converts string into a hash value
def hash_string(s):
    hashed_value = int(hashlib.sha512(s.encode()).hexdigest(), 16)
    short_hash = hashed_value % (2**32) # if higher values like 64, mapping is not accurate
    return short_hash

