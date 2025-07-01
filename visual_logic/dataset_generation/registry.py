DATASET_GENERATORS = {}


def register_dataset(name):
    def decorator(fn):
        DATASET_GENERATORS[name] = fn
        return fn

    return decorator
