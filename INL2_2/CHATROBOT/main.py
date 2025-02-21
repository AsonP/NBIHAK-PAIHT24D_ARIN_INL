# 🚀 Chatbot with RAG (Retrieval-Augmented Generation)
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_community.document_loaders import TextLoader

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY is missing! Check your .env file.")

# ✅ Load metadata from JSON
def load_metadata(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"🚨 Error: Metadata file not found at {metadata_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"🚨 Error: Invalid JSON format in {metadata_path}. Details: {e}")
        return {}

# ✅ Load restaurant information (text)
text_file_path = "data/text_data.txt"
loader = TextLoader(text_file_path)
documents = loader.load()
text_data = documents[0].page_content if documents else ""

# ✅ Load metadata
metadata_file_path = "data/metadata.json"
metadata = load_metadata(metadata_file_path)
print("✅ Metadata loaded successfully!")

# ✅ Create a system prompt using SystemMessagePromptTemplate
system_prompt_template = SystemMessagePromptTemplate.from_template(f"""
🏢 **Welcome to CosmoByte Diner!** 🚀

📍 **Location & Contact Info:**
{metadata['restaurant']['location']['address']}
For reservations, call: {metadata['restaurant']['contact']['phone']}
Visit our website: {metadata['restaurant']['contact']['website']}

🛠 **How I Work:** 
I am an AI assistant for CosmoByte Diner. I answer questions based on the restaurant information, menu, and services. 

📑 **Restaurant Details:**
{text_data}

🔹 **Rules:**  
- If someone asks about the location, use metadata.
- If someone asks for the menu, provide details from metadata.
- If someone asks about reservations, use metadata.
- If something is missing, say **"I don't have that information"** instead of guessing.
""")

# ✅ Create a ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages([
    system_prompt_template,
    HumanMessagePromptTemplate.from_template("{input}")
])

# ✅ Create an OpenAI-powered chat model
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# ✅ Dictionary to store session-based chat history
session_histories = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a given session."""
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

# ✅ Main chatbot loop
def chatbot_loop():
    print("🤖 **Chatbot initialized. Ask me anything!**")

    session_id = "default_session"  # Ensures history tracking

    # ✅ Create a RAG-based chat pipeline with correct session handling
    chat_memory = RunnableWithMessageHistory(
        runnable=chat_prompt_template | llm,
        get_session_history=get_session_history,
        input_key="input",
        history_key="history"
    )

    while True:
        user_input = input("💬 You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 **Chatbot:** Goodbye! Have a great day!")
            break

        try:
            # ✅ Invoke chat with session_id in the config
            response = chat_memory.invoke(
                {
                    "input": user_input,
                },
                config={"configurable": {"session_id": session_id}}
            )

            # ✅ Extract only the text content (fix metadata issue)
            if hasattr(response, "content"):
                response_text = response.content  # Extracting only the AI's message text
            else:
                response_text = str(response)  # Fallback for unexpected formats

            print(f"🤖 **Chatbot:** {response_text}")
        except Exception as e:
            print(f"🚨 **Error processing:** {e}")
            print(f"🔎 **User Input:** {user_input}")

# ✅ Run chatbot if script is executed directly
if __name__ == "__main__":
    chatbot_loop()