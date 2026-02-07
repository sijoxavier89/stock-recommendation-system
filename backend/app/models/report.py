from pydantic import BaseModel


class Report(BaseModel):
    company: str
    year: int
    path: str
