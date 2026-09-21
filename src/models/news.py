from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class NewsArticle(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    title: str
    content: str
    summary: str | None = None
    published_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    source: str
    author: str | None = None
    metadata: dict = Field(default_factory=dict)

    # Processing status
    scored: bool = False
    score_id: UUID | None = None


class SearchResult(BaseModel):
    url: HttpUrl
    title: str
    snippet: str
    published_at: datetime | None = None
    source: str
