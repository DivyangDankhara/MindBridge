from datetime import datetime, timezone

from pydantic import BaseModel


class MissionExperience(BaseModel):
    timestamp: str
    intent_task: str
    intent_goal: str
    success: bool
    attempts: int
    final_plan_summary: str
    failure_reasons: list[str]
    tools_used: list[str]
    result_summary: str

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()
