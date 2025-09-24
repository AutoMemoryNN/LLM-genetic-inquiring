import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" o "local"


settings = Settings()
