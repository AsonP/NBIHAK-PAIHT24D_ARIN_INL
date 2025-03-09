import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr

# Ladda miljövariabler från .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Konfiguration
CACHE_FILE = "data/jevees_cache.json"
BACKGROUND_FILE = "data/jevees_background.json"
INDEX_FILE = "data/index.faiss"
EMBEDDING_DIM = 768

# Initiera SentenceTransformer för embeddings
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Ladda JSON-data
background_data = json.load(open(BACKGROUND_FILE, encoding="utf-8"))
traits = background_data.get("traits", {})

def format_traits(traits):
    return "\n".join([f"- {key.capitalize()}: {value}" for key, value in traits.items()])

def format_chat_history(history):
    if not history:
        return "Ingen tidigare konversation."
    formatted = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted += f"Användaren: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"Jevees: {msg.content}\n"
    return formatted.strip()

def validate_json_file(filepath, default_data=None):
    if not os.path.exists(filepath):
        print(f"{filepath} saknas – skapar en ny...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_data if default_data else {}, f, ensure_ascii=False, indent=4)
        return default_data if default_data else {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) and not isinstance(data, list):
                print(f"{filepath} var inte ett giltigt JSON-objekt – återställs.")
                return default_data if default_data else {}
            return data
    except json.JSONDecodeError:
        print(f"{filepath} var korrupt – skapar en ny...")
        return default_data if default_data else {}

def validate_faiss_index(index_path, embedding_dim):
    if not os.path.exists(index_path):
        print(f"{index_path} saknas – skapar en ny FAISS-index...")
        index = faiss.IndexFlatL2(embedding_dim)
        faiss.write_index(index, index_path)
        return index
    try:
        index = faiss.read_index(index_path)
        print(f"FAISS-indexet {index_path} laddat framgångsrikt.")
        print(f"Embedding-dimension i index: {index.d}")
        return index
    except Exception as e:
        print(f"Fel vid läsning av {index_path}: {e} – skapar ett nytt FAISS-index...")
        index = faiss.IndexFlatL2(embedding_dim)
        faiss.write_index(index, index_path)
        return index

def retrieve_relevant_text(query, index, texts, top_k=2):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    if len(indices) == 0 or len(indices[0]) == 0:
        return ["Jag har tyvärr ingen specifik information om det, men jag kan ge en generell rekommendation."]
    return [texts[i] for i in indices[0] if i < len(texts)]

# Validera filer
cache = validate_json_file(CACHE_FILE, {})
background_data = validate_json_file(BACKGROUND_FILE, {"background": [], "traits": {}})
index = validate_faiss_index(INDEX_FILE, EMBEDDING_DIM)

# Skapa embeddings och FAISS-index
texts = [entry["content"] for entry in background_data["background"]]
embeddings = model.encode(texts, convert_to_numpy=True)
print(f"Embedding-dimension från modell: {embeddings.shape[1]}")
if index.d != embeddings.shape[1]:
    print(f"Dimension mismatch! Index: {index.d}, Embeddings: {embeddings.shape[1]} – Skapar nytt index.")
    index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

# Skapa minne med InMemoryChatMessageHistory
chat_history = InMemoryChatMessageHistory()

# Dynamisk prompt
prompt_template = PromptTemplate(
    input_variables=["chat_history", "context", "question", "traits"],
    template="""
    📌 Du är Jevees, en sofistikerad butler med följande egenskaper:
    {traits}
    
    📜 Tidigare konversation:
    {chat_history}
    
    📚 Historik:
    {context}
    
    Om användaren ställer en direkt följdfråga, håll svaret kortfattat och undvik upprepning.
    Svara på samma språk som frågan är ställd på (t.ex. svenska för svenska frågor, engelska för engelska frågor).
    
    Baserat på detta, svara på frågan:
    "{question}"
    """
)

# Skapa LLM och bas-kedja
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
base_chain = prompt_template | llm

# Skapa en kedja med minneshantering
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def chat_with_jevees(user_input, history):
    if not user_input:
        return history
    
    # Om användaren skriver "exit", "quit" eller "avsluta", avsluta konversationen
    if user_input.lower() in ["exit", "quit", "avsluta"]:
        history.append(("Du", user_input))
        history.append(("Jevees", "Ha en fortsatt trevlig dag!"))
        return history
    
    # Hämta relevant kontext
    relevant_info = retrieve_relevant_text(user_input, index, texts)
    context = "\n".join(relevant_info)
    
    # Formatera chat-historiken (begränsa till 5 meddelanden)
    chat_history_formatted = format_chat_history(chat_history.messages[:5])
    
    # Kör kedjan med historik
    response = chain_with_history.invoke(
        {
            "chat_history": chat_history_formatted,
            "context": context,
            "question": user_input,
            "traits": format_traits(traits)
        },
        config={"configurable": {"session_id": "default"}}
    )
    
    # Spara i minnet
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response.content)
    
    # Spara i cache
    cache[user_input] = response.content
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    
    # Uppdatera Gradio-historiken
    history.append(("Du", user_input))
    history.append(("Jevees", response.content))
    return history

# Skapa Gradio-gränssnitt med anpassad bakgrundsfärg
with gr.Blocks(
    title="Jevees – Din Sofistikerade Butler",
    theme=gr.themes.Default(primary_hue="blue").set(
        body_background_fill="#f5f5f5"  # Ljusgrå bakgrund för bättre läsbarhet
    )
) as demo:
    gr.Markdown("# 📌 Välkommen till Jevees!", container=False)
    gr.Markdown("Ställ en fråga till Jevees, din personliga butler. Han svarar på samma språk som din fråga!", container=False)
    
    chatbot = gr.Chatbot(label="💬 Konversation")
    msg = gr.Textbox(label="✍️ Skriv din fråga här", placeholder="Vad vill du veta?")
    clear = gr.Button("🗑️ Rensa konversationen")
    
    # Koppla funktionen till textboxen
    msg.submit(chat_with_jevees, inputs=[msg, chatbot], outputs=[chatbot])
    
    # Rensa historiken
    clear.click(lambda: [], outputs=[chatbot])

# Starta gränssnittet
demo.launch()