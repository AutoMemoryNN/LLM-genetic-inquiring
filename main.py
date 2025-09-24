# main.py
from llm_manager import LLMManager

def main():
    # Combos configurables
    manager = LLMManager(provider="openai", chat_model="cheap", embed_model="cheap")

    llm = manager.get_chat_llm()
    emb = manager.get_embeddings()

    # Usar LLM
    q = "Dame 2 ideas creativas para reutilizar basura espacial"
    print("Respuesta:\n", llm.invoke(q))

    # Usar embeddings
    vector = emb.embed_query(q)
    print("Embedding dim:", len(vector))

if __name__ == "__main__":
    main()
