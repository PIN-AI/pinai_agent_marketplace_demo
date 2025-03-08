import os
import time
import re
import streamlit as st

from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    app_name: str = "PIN AI Agent Marketplace"
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    port: int = int(os.getenv("PORT", "8000"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()


def setup_streamlit():
    # Streamlit page config
    st.set_page_config(page_title="PIN AI Agent Marketplace", page_icon="🤖")
    st.title("🤖 PIN AI Agent Marketplace")
    st.caption("An example of two-way agent-to-agent chat to complete a task")


# Streamed response emulator
def response_generator(response):
    # Split response into words while preserving whitespace and newlines
    words = re.split(r'(\s+)', response)
    for word in words:
        yield word
        time.sleep(0.01) 
