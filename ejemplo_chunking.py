# ejemplo_chunking.py
from llm_manager import LLMManager
import numpy as np


def main():
    # Crear el manager
    manager = LLMManager(provider="openai", chat_model="cheap", embed_model="cheap")

    # Ejemplo de texto largo (simulando un artículo o documento)
    texto_largo = """
    La inteligencia artificial (IA) es una rama de la informática que se centra en la creación de sistemas 
    capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas pueden aprender, 
    razonar, percibir, comprender el lenguaje natural y resolver problemas complejos.
    
    Los algoritmos de aprendizaje automático son fundamentales en la IA moderna. Permiten a las máquinas 
    aprender patrones a partir de datos sin ser programadas explícitamente para cada tarea específica. 
    Esto incluye técnicas como redes neuronales, árboles de decisión, y algoritmos de clustering.
    
    Las redes neuronales artificiales están inspiradas en el funcionamiento del cerebro humano. Consisten 
    en capas de nodos interconectados que procesan información de manera paralela. Las redes neuronales 
    profundas, con múltiples capas ocultas, han revolucionado campos como el reconocimiento de imágenes, 
    el procesamiento de lenguaje natural y la generación de contenido.
    
    El procesamiento de lenguaje natural (NLP) es una subdisciplina de la IA que se enfoca en la interacción 
    entre computadoras y lenguaje humano. Los modelos de lenguaje como GPT y BERT han demostrado capacidades 
    impresionantes en tareas como traducción, resumen de texto, y generación de contenido coherente.
    
    Los sistemas de recomendación utilizan IA para sugerir productos, servicios o contenido relevante a los 
    usuarios basándose en sus preferencias y comportamientos pasados. Estos sistemas son fundamentales en 
    plataformas como Netflix, Amazon, y Spotify.
    
    La visión por computadora permite a las máquinas interpretar y analizar contenido visual. Aplicaciones 
    incluyen reconocimiento facial, detección de objetos, diagnóstico médico por imágenes, y vehículos autónomos.
    
    Los desafíos éticos en IA incluyen sesgo algorítmico, privacidad de datos, transparencia en la toma de 
    decisiones, y el impacto en el empleo. Es crucial desarrollar IA responsable que beneficie a toda la sociedad.
    
    El futuro de la IA promete avances en áreas como la inteligencia artificial general (AGI), computación 
    cuántica aplicada a IA, y la integración más profunda de IA en la vida cotidiana y los procesos industriales.
    """

    print("=== Análisis del texto ===")
    tokens_estimados = manager.get_token_count_estimate(texto_largo)
    print(f"Longitud del texto: {len(texto_largo)} caracteres")
    print(f"Tokens estimados: {tokens_estimados}")
    print()

    # Ejemplo 1: Procesar texto largo con chunking
    print("=== Ejemplo 1: Procesamiento con chunking ===")
    max_tokens = 200  # Límite pequeño para demostrar el chunking
    embeddings_list, chunks = manager.text_to_embedding_with_chunking(
        texto_largo, max_tokens=max_tokens, overlap=50
    )

    print(f"Texto dividido en {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        tokens_chunk = manager.get_token_count_estimate(chunk)
        print(f"Chunk {i}: {tokens_chunk} tokens, {len(chunk)} caracteres")
        print(f"Inicio: '{chunk[:50]}...'")
        print(f"Embedding shape: {embeddings_list[i - 1].shape}")
        print()

    # Ejemplo 2: Combinar embeddings de chunks
    print("=== Ejemplo 2: Combinación de embeddings ===")
    embedding_combinado_mean = manager.combine_embeddings(
        embeddings_list, method="mean"
    )
    embedding_combinado_max = manager.combine_embeddings(embeddings_list, method="max")

    print(f"Embedding combinado (mean) shape: {embedding_combinado_mean.shape}")
    print(f"Embedding combinado (max) shape: {embedding_combinado_max.shape}")
    print()

    # Ejemplo 3: Comparar con embedding de texto truncado
    print("=== Ejemplo 3: Comparación con texto truncado ===")
    embedding_truncado = manager.text_to_single_embedding(texto_largo, max_tokens=500)
    print(f"Embedding de texto truncado shape: {embedding_truncado.shape}")

    # Calcular similitud entre enfoques
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_mean_vs_truncado = cosine_similarity(
        embedding_combinado_mean, embedding_truncado
    )
    sim_max_vs_truncado = cosine_similarity(embedding_combinado_max, embedding_truncado)

    print(f"Similitud (chunks mean vs truncado): {sim_mean_vs_truncado:.4f}")
    print(f"Similitud (chunks max vs truncado): {sim_max_vs_truncado:.4f}")
    print()

    # Ejemplo 4: Diferentes tamaños de chunk
    print("=== Ejemplo 4: Diferentes tamaños de chunk ===")
    for max_tokens_test in [100, 300, 500]:
        embeddings_test, chunks_test = manager.text_to_embedding_with_chunking(
            texto_largo, max_tokens=max_tokens_test
        )
        print(
            f"Con {max_tokens_test} tokens máximo: {len(chunks_test)} chunks generados"
        )

    print("\n=== Recomendaciones de uso ===")
    print("- Para textos cortos (< 8000 tokens): usar text_to_single_embedding()")
    print("- Para textos largos: usar text_to_embedding_with_chunking()")
    print("- Para combinar chunks: 'mean' preserva información promedio")
    print("- Para combinar chunks: 'max' preserva características más prominentes")
    print("- Ajustar max_tokens según el modelo de embeddings utilizado")


if __name__ == "__main__":
    main()
