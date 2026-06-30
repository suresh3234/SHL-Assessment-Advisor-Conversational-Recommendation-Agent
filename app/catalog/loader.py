import json
import os
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field, computed_field

class CatalogItem(BaseModel):
    entity_id: str
    name: str
    url: HttpUrl = Field(validation_alias="link")
    description: Optional[str] = None
    job_levels: List[str] = []
    languages: List[str] = []
    duration: Optional[str] = None
    remote: Optional[str] = None
    adaptive: Optional[str] = None
    keys: List[str] = []

    @computed_field
    @property
    def test_type(self) -> str:
        return self.keys[0] if self.keys else "Knowledge & Skills"

def load_catalog(filepath: Optional[str] = None) -> List[CatalogItem]:
    """
    Loads and validates catalog.json into a list of CatalogItem records.
    """
    if not filepath:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, "catalog.json")
        
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, "r", encoding="utf-8") as f:
        # strict=False allows literal control characters (like newlines) in strings
        data = json.load(f, strict=False)
        
    return [CatalogItem.model_validate(item) for item in data]

