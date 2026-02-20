import json
import re

from memory.schema import MissionExperience
from memory.store import EPISODIC_FILE, ensure_episodic_file


_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.lower()))


def _keywords(text: str) -> set[str]:
    return {token for token in _WORD_PATTERN.findall(text.lower()) if len(token) > 2}


def _similarity_score(exp: MissionExperience, task: str, goal: str) -> float:
    task_norm = _normalize_text(task)
    goal_norm = _normalize_text(goal)
    exp_task_norm = _normalize_text(exp.intent_task)
    exp_goal_norm = _normalize_text(exp.intent_goal)

    score = 0.0
    if task_norm and task_norm == exp_task_norm:
        score += 4.0
    elif task_norm and (task_norm in exp_task_norm or exp_task_norm in task_norm):
        score += 2.0

    if goal_norm and goal_norm == exp_goal_norm:
        score += 4.0
    elif goal_norm and (goal_norm in exp_goal_norm or exp_goal_norm in goal_norm):
        score += 2.0

    query_terms = _keywords(f"{task} {goal}")
    exp_terms = _keywords(f"{exp.intent_task} {exp.intent_goal} {exp.result_summary}")
    score += float(len(query_terms.intersection(exp_terms)))

    return score


def load_all_experiences() -> list[MissionExperience]:
    ensure_episodic_file()
    experiences: list[MissionExperience] = []

    with EPISODIC_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                experiences.append(MissionExperience.model_validate(json.loads(payload)))
            except Exception:
                continue

    return experiences


def find_similar_experiences(task: str, goal: str, limit: int = 3) -> list[MissionExperience]:
    if limit <= 0:
        return []

    scored: list[tuple[float, MissionExperience]] = []
    for exp in load_all_experiences():
        score = _similarity_score(exp, task, goal)
        if score <= 0:
            continue
        scored.append((score, exp))

    scored.sort(key=lambda item: (item[0], item[1].success, item[1].timestamp), reverse=True)
    return [item[1] for item in scored[:limit]]
