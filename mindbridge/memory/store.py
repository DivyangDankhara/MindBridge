from pathlib import Path
from typing import Any

from memory.schema import MissionExperience


EPISODIC_FILE = Path(__file__).resolve().parent / "episodic.jsonl"


def ensure_episodic_file() -> Path:
    EPISODIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    EPISODIC_FILE.touch(exist_ok=True)
    return EPISODIC_FILE


def save_experience(exp: MissionExperience | dict[str, Any]) -> None:
    record = exp if isinstance(exp, MissionExperience) else MissionExperience.model_validate(exp)
    file_path = ensure_episodic_file()

    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


ensure_episodic_file()
