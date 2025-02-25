# 🤖 Chatbot med RAG (Retrieval-Augmented Generation)
Denna applikation är en chatbot som använder Retrieval-Augmented Generation (RAG) för att svara på användarfrågor baserat på specifik information från en restaurang, CosmoByte Diner. Chatboten är utvecklad med moderna AI-verktyg och ramverk för att ge snabba och precisa svar.

# Teknologier och verktyg
För att bygga denna applikation har vi använt följande tekniker och bibliotek:

* LangChain: För att hantera konversationer och integrera språkmodeller.
* OpenAI (GPT-4): Som språkmodell för att generera svar på användarfrågor.
* TextLoader: För att ladda textdata från filer.
*InMemoryChatMessageHistory: För att hantera konversationshistorik under sessioner.

# Datakällor
Chatboten baserar sina svar på två datakällor:

* metadata.json: Innehåller metadata om restaurangen, såsom adress, kontaktinformation och meny.
*text_data.txt: Innehåller detaljerad information om restaurangen, såsom beskrivningar, specialerbjudanden och öppettider.

# Hur applikationen fungerar
Användaren ställer en fråga via terminalen.

* Chatboten söker igenom textdata för att hitta relevant information.
* Språkmodellen (GPT-4) genererar ett svar baserat på den hittade informationen.
* Svaret visas i terminalen tillsammans med tidigare frågor och svar.


# 📄 Licens
Detta projekt är en skoluppgift och används endast i utbildningssyfte.

