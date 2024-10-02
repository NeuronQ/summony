def separate_prefixed(d: dict, prefix: str) -> tuple[dict, dict]:
    params_from_kwargs = {}
    left_kwargs = {}
    for k, v in d.items():
        if k.startswith(prefix):
            params_from_kwargs[k[len(prefix) :]] = v
        else:
            left_kwargs[k] = v
    return params_from_kwargs, left_kwargs


def _dict_to_deep_frozenset(d, _depth=0):
    if _depth > 42:
        return d
    return frozenset(
        {
            k: _dict_to_deep_frozenset(v, _depth + 1) if isinstance(v, dict) else v
            for k, v in d.items()
        }.items()
    )


class HashableDict(dict):
    def __hash__(self):
        return hash(_dict_to_deep_frozenset(self))
