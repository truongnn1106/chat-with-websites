from pydantic import BaseModel


class ChatRequest(BaseModel):
    url: str = "https://vucar.vn/"
    message: str
    chat_history: list = []


class ChatResponse(BaseModel):
    answer: str
    chat_history: list
