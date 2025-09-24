# llm_manager.py
import numpy as np
import re
from typing import List, Tuple
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

    # --- Métodos de procesamiento de texto con límites ---
    def text_to_embedding_with_chunking(
        self, text: str, max_tokens: int = 8000, overlap: int = 100
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Convierte un texto largo a embeddings dividiéndolo en chunks que respeten
        los límites de tokens del modelo.

        Args:
            text (str): El texto largo a procesar
            max_tokens (int): Número máximo de tokens por chunk (default: 8000)
            overlap (int): Número de caracteres de solapamiento entre chunks

        Returns:
            Tuple[List[np.ndarray], List[str]]:
                - Lista de embeddings (uno por chunk)
                - Lista de chunks de texto procesados
        """
        # Dividir el texto en chunks
        chunks = self._split_text_into_chunks(text, max_tokens, overlap)

        # Obtener embeddings para cada chunk
        embeddings_list = []
        embeddings_model = self.get_embeddings()

        for chunk in chunks:
            if chunk.strip():  # Solo procesar chunks no vacíos
                embedding_vector = embeddings_model.embed_query(chunk)
                embeddings_list.append(np.array(embedding_vector, dtype=np.float32))

        return embeddings_list, chunks

    def _split_text_into_chunks(
        self, text: str, max_tokens: int, overlap: int
    ) -> List[str]:
        """
        Divide un texto en chunks respetando límites de tokens.
        Usa una aproximación: ~4 caracteres = 1 token para español/inglés.

        Args:
            text (str): Texto a dividir
            max_tokens (int): Máximo tokens por chunk
            overlap (int): Caracteres de solapamiento

        Returns:
            List[str]: Lista de chunks de texto
        """
        # Aproximación: 4 caracteres ≈ 1 token para texto en español/inglés
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token

        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calcular el final del chunk
            end = start + max_chars

            if end >= len(text):
                # Último chunk
                chunks.append(text[start:])
                break

            # Buscar un punto de corte natural (espacio, punto, etc.)
            chunk_text = text[start:end]

            # Intentar cortar en una oración completa
            last_sentence = chunk_text.rfind(". ")
            if (
                last_sentence > max_chars * 0.7
            ):  # Si el corte está en al menos 70% del chunk
                end = start + last_sentence + 2
            else:
                # Si no, cortar en un espacio
                last_space = chunk_text.rfind(" ")
                if (
                    last_space > max_chars * 0.8
                ):  # Si el espacio está en al menos 80% del chunk
                    end = start + last_space

            chunks.append(text[start:end])

            # Calcular el siguiente inicio con solapamiento
            start = end - overlap
            if start < 0:
                start = 0

        return chunks

    def get_token_count_estimate(self, text: str) -> int:
        """
        Estima el número de tokens en un texto.
        Usa una aproximación simple: ~4 caracteres = 1 token.

        Args:
            text (str): Texto a analizar

        Returns:
            int: Estimación del número de tokens
        """
        # Remover espacios extra y contar caracteres significativos
        clean_text = re.sub(r"\s+", " ", text.strip())
        return len(clean_text) // 4

    def text_to_single_embedding(self, text: str, max_tokens: int = 8000) -> np.ndarray:
        """
        Convierte texto a un solo embedding. Si el texto es muy largo,
        lo trunca o toma solo la parte inicial.

        Args:
            text (str): Texto a convertir
            max_tokens (int): Límite máximo de tokens

        Returns:
            np.ndarray: Embedding del texto (posiblemente truncado)
        """
        # Verificar si el texto excede el límite
        if self.get_token_count_estimate(text) > max_tokens:
            # Truncar el texto
            max_chars = max_tokens * 4
            text = text[:max_chars]
            # Cortar en un espacio para no partir palabras
            last_space = text.rfind(" ")
            if last_space > 0:
                text = text[:last_space]

        embeddings_model = self.get_embeddings()
        embedding_vector = embeddings_model.embed_query(text)
        return np.array(embedding_vector, dtype=np.float32)

    def combine_embeddings(
        self, embeddings_list: List[np.ndarray], method: str = "mean"
    ) -> np.ndarray:
        """
        Combina múltiples embeddings en uno solo.

        Args:
            embeddings_list (List[np.ndarray]): Lista de embeddings
            method (str): Método de combinación ('mean', 'max', 'sum')

        Returns:
            np.ndarray: Embedding combinado
        """
        if not embeddings_list:
            raise ValueError("La lista de embeddings está vacía")

        embeddings_matrix = np.array(embeddings_list)

        if method == "mean":
            return np.mean(embeddings_matrix, axis=0)
        elif method == "max":
            return np.max(embeddings_matrix, axis=0)
        elif method == "sum":
            return np.sum(embeddings_matrix, axis=0)
        else:
            raise ValueError(
                f"Método '{method}' no soportado. Use: 'mean', 'max', 'sum'"
            )
