from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class Dimensions(BaseModel):
    height: int = 0
    width: int = 0


class Owner(BaseModel):
    id: str


class ThumbnailResource(BaseModel):
    config_height: int
    config_width: int
    src: HttpUrl


class CaptionNode(BaseModel):
    text: str = ""


class CaptionEdge(BaseModel):
    node: CaptionNode = Field(default_factory=CaptionNode)


class CaptionContainer(BaseModel):
    edges: List[CaptionEdge] = []


class CountContainer(BaseModel):
    count: int = 0


class InstagramPost(BaseModel):
    typename: str = Field(alias="__typename", default="InstagramPost")
    id: str
    shortcode: Optional[str] = None
    comments_disabled: bool = False
    dimensions: Dimensions = Field(default_factory=Dimensions)
    display_url: Optional[HttpUrl] = None
    # Handle the complex nested objects with defaults
    edge_media_preview_like: CountContainer = Field(default_factory=CountContainer)
    edge_media_to_comment: CountContainer = Field(default_factory=CountContainer)
    edge_media_to_caption: CaptionContainer = Field(default_factory=CaptionContainer)
    is_video: bool = False
    media_preview: Optional[str] = None
    owner: Optional[Owner] = None

    # allow_inf_nan=True helps if timestamps are weirdly formatted
    taken_at_timestamp: Optional[datetime] = None

    thumbnail_src: Optional[HttpUrl] = None
    thumbnail_resources: List[ThumbnailResource] = []
    gating_info: Optional[Dict[str, Any]] = None

    class Config:
        # This allows the model to accept both the alias and the property name
        allow_population_by_field_name = True

    @property
    def caption_text(self) -> str:
        """Helper to extract the first caption text easily."""
        if self.edge_media_to_caption.edges:
            return self.edge_media_to_caption.edges[0].node.text
        return ""


class Influencer(BaseModel):
    name: str
    followers: int
    followees: int
    posts: int
    url: Optional[str] = None
    category: Optional[str] = None
    bio: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    profile_pic_url: Optional[str] = None
    text_desc: Optional[str] = None
