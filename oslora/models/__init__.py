"""Model adapters (see base.py). Imported lazily: each pulls in its own model code."""

MODELS = {"moss_image": "MOSS-Image", "krea2": "Krea 2"}


def get_adapter(key: str):
    if key == "moss_image":
        from .moss.adapter import MossAdapter

        return MossAdapter()
    if key == "krea2":
        from .krea2.adapter import Krea2Adapter

        return Krea2Adapter()
    raise ValueError(f"지원하지 않는 모델: {key}")
