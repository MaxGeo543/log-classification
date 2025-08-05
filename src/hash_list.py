import hashlib
import base64

def hash_list_to_string(str_list, length):
    """
    Hashes a list of strings into a fixed‐length string.
    
    :param str_list: List of input strings.
    :param length: Desired length of output string.
    :return: A string of exactly `length` characters.
    """
    # 1) Create hash and feed in each string
    h = hashlib.sha256()
    for s in str_list:
        h.update(s.encode('utf-8'))
    
    # 2) Base64‐encode the raw digest, URL‐safe, strip padding
    b64 = base64.urlsafe_b64encode(h.digest()).decode('ascii').rstrip('=')
    
    # 3) If that’s already long enough, just truncate
    if len(b64) >= length:
        return b64[:length]
    
    # 4) Otherwise, extend by re-hashing with a salt
    #    until we have enough characters
    extra = b64
    counter = 0
    while len(extra) < length:
        counter += 1
        h2 = hashlib.sha256()
        # use the original digest + a counter as “salt”
        h2.update(h.digest() + counter.to_bytes(4, 'big'))
        extra += base64.urlsafe_b64encode(h2.digest()).decode('ascii').rstrip('=')
    
    return extra[:length]

