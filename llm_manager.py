# llm_manager.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings


class LLMManager:
    def __init__(self, provider="openai", chat_model="cheap", embed_model="cheap"):
        self.provider = provider
        self.chat_model = chat_model
        self.embed_model = embed_model

    # --- Selección de Chat LLM ---
    def get_chat_llm(self):
        if self.provider == "openai":
            if self.chat_model == "cheap":
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    api_key=settings.OPENAI_API_KEY,
                )  # barato
            elif self.chat_model == "advanced":
                return ChatOpenAI(
                    model="gpt-4.1", temperature=0.7, api_key=settings.OPENAI_API_KEY
                )  # más avanzado
        elif self.provider == "local":
            # aquí pondrías tu modelo local HuggingFace si quieres
            raise NotImplementedError("Local provider aún no implementado")
        else:
            raise ValueError("Proveedor no soportado")

    # --- Selección de Embeddings ---
    def get_embeddings(self):
        if self.provider == "openai":
            if self.embed_model == "cheap":
                return OpenAIEmbeddings(
                    model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
                )  # barato
            elif self.embed_model == "advanced":
                return OpenAIEmbeddings(
                    model="text-embedding-3-large", api_key=settings.OPENAI_API_KEY
                )  # más caro/preciso
        elif self.provider == "local":
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )  # rápido y barato local
        else:
            raise ValueError("Proveedor no soportado")
