import os
import json
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 📌 Ladda miljövariabler från .env-filen
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY saknas! Kontrollera din .env-fil.")

# Tvinga användning av CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Inaktivera GPU

# Dynamisk sökväg
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Läs metadata.json
metadata_path = os.path.join(DATA_DIR, "metadata.json")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Filen {metadata_path} finns inte!")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Läs text_data.txt
text_data_path = os.path.join(DATA_DIR, "text_data.txt")
if not os.path.exists(text_data_path):
    raise FileNotFoundError(f"Filen {text_data_path} finns inte!")
with open(text_data_path, "r", encoding="utf-8") as f:
    text_data = f.read()

# Hämta exempelmeningar från JSON
examples = metadata.get("examples", [])

# Systemprompt dynamiskt genererad
system_prompt = "You are an AI assistant for CosmoByte Diner. Answer user questions based on the restaurant's metadata.\n\n"
if examples:
    system_prompt += "Here are some example questions you can answer:\n"
    for ex in examples:
        system_prompt += f"- {ex}\n"

# Textsplitter och embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_text(text_data)
documents = [Document(page_content=chunk) for chunk in chunks]

# Använd CPU för HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}  # Tvinga användning av CPU
)
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever()

# Konfigurera LLM och RAG pipeline
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)
rag_chain = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
]) | llm

# Egen hantering av konversationshistorik
conversation_history = []

def chat_interface(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Skapa prompt med historik
    prompt = system_prompt
    for message in conversation_history:
        prompt += f"{message['role'].capitalize()}: {message['content']}\n"
    
    # Generera svar
    response = rag_chain.invoke({"input": prompt})
    conversation_history.append({"role": "assistant", "content": response.content})
    
    # Returnera historik i det korrekta formatet för Gradio
    return [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history]

# Gradio UI med mörk bakgrund
css = """
body, .gradio-container {
    background-color: #2E2E2E !important;
    color: white !important;
}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🤖 CosmoByte Diner Chatbot")
    chatbot = gr.Chatbot(type="messages")  # Ange type='messages'
    msg = gr.Textbox(placeholder="Ask me anything about the restaurant...")
    clear = gr.Button("Clear Chat")
    
    def clear_history():
        global conversation_history
        conversation_history = []
        return []
    
    msg.submit(chat_interface, inputs=msg, outputs=chatbot)
    clear.click(clear_history, outputs=chatbot)

# Starta Gradio server
if __name__ == "__main__":
    demo.launch()  # Kör lokalt utan offentlig länk