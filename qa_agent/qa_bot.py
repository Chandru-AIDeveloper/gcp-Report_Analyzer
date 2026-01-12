import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOllama(
    model="phi3:mini",
    temperature=0.2
)

# GLOBAL long-term memory
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_texts(["Initial memory store"], embedding_model)
    vectorstore.save_local("faiss_index")

# NEW: short-term memory per session
session_memory = {}

def get_answer(question: str, context: str, session_id: str) -> str:
    try:
        # Load session-specific memory
        if session_id not in session_memory:
            session_memory[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        # Retrieve previous messages
        chat_history_messages = session_memory[session_id].chat_memory.messages
        chat_history_text = ""
        if chat_history_messages:
            history_lines = []
            for msg in chat_history_messages[-6:]:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content}")
            chat_history_text = "\n".join(history_lines)

        docs = vectorstore.similarity_search(question, k=2)
        long_term_context = "\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template(
       
       """You are an **Insight-Based Chat Assistant**.

You answer user questions **only using the provided insights**.
The insights are already analyzed and verified.

---------------------------------------------------------
STRICT RULES (DO NOT VIOLATE)
---------------------------------------------------------
1. Use ONLY the provided insights as your knowledge source
2. DO NOT invent, assume, or estimate any data
3. DO NOT refer to raw data, JSON, or source files
4. DO NOT introduce new metrics or categories
5. If the answer is not present in the insights, respond clearly:
   "The available insights do not contain enough information to answer this."
6. Be concise, accurate, and professional
7. Highlight **important keywords** when helpful
8.If the answer is not clear for the user query , First thing you fully analyzed then if the data is presented give the answer correctly.

---------------------------------------------------------
CHAT HISTORY (Continuity):
---------------------------------------------------------
{chat_history}

---------------------------------------------------------
RELEVANT PAST KNOWLEDGE (Long-term Memory):
---------------------------------------------------------
{long_memory}

---------------------------------------------------------
INSIGHTS CONTEXT:
---------------------------------------------------------
{context}
---------------------------------------------------------

---------------------------------------------------------
USER QUESTION:
---------------------------------------------------------
{question}
---------------------------------------------------------

---------------------------------------------------------
HOW TO RESPOND:
---------------------------------------------------------
- Answer directly and clearly
- Use insight-based reasoning only
- For improvement-related questions, use Suggestions
- For distribution/count questions, use Chart Data
- Do NOT repeat the entire insights unless necessary
- Do NOT add explanations beyond available insights

---------------------------------------------------------
FINAL ANSWER (Plain Text Only):
---------------------------------------------------------
"""
        )
        formatted_prompt = prompt.format(
            chat_history=chat_history_text or "No previous conversation.",
            long_memory=long_term_context or "No relevant long-term memory.",
            context=context or "No external context provided.",
            question=question
        )

        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Save messages in THIS session only
        session_memory[session_id].chat_memory.add_user_message(question)
        session_memory[session_id].chat_memory.add_ai_message(answer)

        # Update FAISS
        vectorstore.add_texts([f"Q: {question}\nContext: {context}\nA: {answer}"])
        vectorstore.save_local("faiss_index")

        return answer

    except Exception as e:
        import traceback
        print(f"Error in get_answer: {traceback.format_exc()}")
        return f"Error generating answer: {str(e)}"
