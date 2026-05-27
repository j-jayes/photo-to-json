from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    book_title: str = Field(..., description="The main title of the source book.")
    publisher: Optional[str] = Field(
        None, description="The organization or entity that published the book."
    )
    publication_year: Optional[int] = Field(None, description="The year of publication.")


class ReportEntry(BaseModel):
    entry_id: int = Field(..., description="The unique index number of the report.")
    authors: List[str] = Field(
        ..., description="List of authors (e.g., ['BORISOV E', 'ZOLOTAREV A']). Leave empty if none."
    )
    title: str = Field(..., description="The primary title of the report or book.")
    translated_title: Optional[str] = Field(
        None, description="The translated title, typically found in parentheses."
    )
    publication_info: str = Field(
        ..., description="Details such as publisher, journal, date, and page numbers."
    )
    is_stitched: bool = Field(
        ..., description="True if this entry was stitched together from multiple images."
    )


class IndexPage(BaseModel):
    entries: List[ReportEntry]


class FinalDocument(BaseModel):
    metadata: DocumentMetadata
    all_reports: List[ReportEntry]
