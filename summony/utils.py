def separate_prefixed(d: dict, prefix: str) -> tuple[dict, dict]:
    params_from_kwargs = {}
    left_kwargs = {}
    for k, v in d.items():
        if k.startswith(prefix):
            params_from_kwargs[k[len(prefix):]] = v
        else:
            left_kwargs[k] = v
    return params_from_kwargs, left_kwargs
