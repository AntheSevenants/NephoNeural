# https://stackoverflow.com/questions/53509826/equivalent-to-pyspark-flatmap-in-python
def flat_map(f, li):
    mapped = map(f, li)
    flattened = flatten_single_dim(mapped)
    yield from flattened

def flatten_single_dim(mapped):
    for item in mapped:
        for subitem in item:
            yield subitem

def unique(array):
    return list(set(array))