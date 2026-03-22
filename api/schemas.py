from pydantic import BaseModel, Field


class SingleRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=100)


class CompareRequest(BaseModel):
    text_a: str
    text_b: str
