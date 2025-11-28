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
    model="gemma:2b",
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
       """
    You are a persistent conversational QA assistant with long-term memory.
    You DO NOT forget previous messages. You track context, store information,
    and answer follow-up questions coherently even if details are not repeated.

    ----------------------------------------------------------------------------
    MEMORY & BEHAVIOR RULES
    ----------------------------------------------------------------------------
    1. Treat the initial report/context as the primary knowledge source.
       You continue to use this data throughout the conversation.
       You must NOT ask the user to resend it unless missing.

    2. Conversation should feel continuous.
       If the user asks something referencing earlier messages (he, she, that, it, previous data),
       you infer meaning based on history and respond intelligently.

    3. When a follow-up question comes:
       - Use chat history first
       - Use stored report/context next
       - Use long_memory ONLY when helpful
       - Respond like you remember the whole discussion

    4. Your response should be:
       ✔ Context-aware
       ✔ Detailed when needed
       ✔ Straight to the point
       ✔ Human-readable (NO JSON unless requested)

    ----------------------------------------------------------------------------
    KNOWLEDGE AVAILABLE
    ----------------------------------------------------------------------------
    🗂 Conversation History (Use to interpret follow-ups)
    {chat_history}

    📄 Report / Core Data (Assume you always have access to this)
    {context}

    🧠 Long-Term Memory from previous interactions
    {long_memory}

    ----------------------------------------------------------------------------
    USER QUESTION
    ----------------------------------------------------------------------------
    {question}

    ----------------------------------------------------------------------------
    FINAL RESPONSE FORMAT (Very Important)
    ----------------------------------------------------------------------------
    - Give a direct answer grounded in report/data
    - Use history to maintain flow of the conversation
    - If the question is open-ended → analyze step-by-step
    - If unclear, ask for clarification instead of guessing

    ### Answer:
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
