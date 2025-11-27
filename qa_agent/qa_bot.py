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
    You are a context-aware conversational AI assistant.
    You think carefully, analyze conversations, recall memory, and provide clear,
    helpful, human-like responses without over-explaining or repeating unnecessary text.

    ---------------------------------------------------------
     Important Behavior Guidelines
    ---------------------------------------------------------
    - Use conversation history to maintain continuity and tone.
    - Use long-term memory only when relevant.
    - Use context to provide accurate, factual, helpful answers.
    - If information is missing, ask gently for clarification instead of assuming.
    - Never repeat the entire memory or history back to the user.
    - Be concise but meaningful — respond like a smart assistant, not a search engine.

    ---------------------------------------------------------
     Available Data
    ---------------------------------------------------------
     Conversation History:
    {chat_history}

     Long-Term Memory:
    {long_memory}

     Contextual Information:
    {context}

    ---------------------------------------------------------
     User Question
    ---------------------------------------------------------
    {question}

    ---------------------------------------------------------
    Your Response (Readable + Helpful)
    ---------------------------------------------------------
    Provide a thoughtful answer that:
      ✓ Addresses the question directly
      ✓ Uses context only when helpful
      ✓ Remains easy to read and natural
      ✓ Includes examples or steps if needed
      ✓ Avoids unnecessary repetition of input

    ### Final Answer:
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
