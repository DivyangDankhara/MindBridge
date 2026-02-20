from pydantic import BaseModel


class Intent(BaseModel):
    task: str
    goal: str
    constraints: str | None = None
    output: str | None = None
