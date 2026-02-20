from pathlib import Path

from intent.schema import Intent


_SUPPORTED_KEYS = {"task", "goal", "constraints", "output"}


def parse_intent(text: str) -> Intent:
    data: dict[str, str | None] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            raise ValueError(f"Invalid intent line (expected KEY: VALUE): {raw_line}")

        key, value = line.split(":", 1)
        key_normalized = key.strip().lower()
        if key_normalized not in _SUPPORTED_KEYS:
            continue

        parsed_value = value.strip()
        data[key_normalized] = parsed_value if parsed_value else None

    missing_required = [field for field in ("task", "goal") if not data.get(field)]
    if missing_required:
        raise ValueError(f"Missing required intent fields: {', '.join(missing_required)}")

    return Intent(**data)


def parse_intent_file(path: str | Path) -> Intent:
    return parse_intent(Path(path).read_text(encoding="utf-8"))
