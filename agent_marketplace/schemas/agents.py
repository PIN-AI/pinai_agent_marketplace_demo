from datetime import datetime

from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
    sender: str
    receiver: str
    timestamp: datetime
    metadata: dict | None = None

class Context(BaseModel):
    history: list[Message]