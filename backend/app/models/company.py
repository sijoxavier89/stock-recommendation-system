from pydantic import BaseModel


class Company(BaseModel):
    ticker: str
    name: str
