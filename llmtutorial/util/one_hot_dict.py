class OneHotDict:

    _keys: tuple[str, ...]
    _values: dict[str, int]

    def __init__(self, keys):
        self._keys = keys
        self._values = {k: 0 for k in keys}

    def set(self, key):
        if key not in self._keys:
            raise KeyError(f"{key} not a valid key")
        for k in self._keys:
            self._values[k] = 0
        self._values[key] = 1

    def __getitem__(self, key):
        return self._values[key]

    def __repr__(self):
        return repr(self._values)
