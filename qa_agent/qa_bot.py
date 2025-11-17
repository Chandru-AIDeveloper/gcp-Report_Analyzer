import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ------------------------------
# Global LLM (Ollama)
# ------------------------------
llm = ChatOllama(
    model="gemma:2b",   # You can change to "llama3", "phi3", etc. depending on what’s installed
    temperature=0.3
)

# ------------------------------
# Short-term memory (per session)
# ------------------------------
short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ------------------------------
# Long-term memory (FAISS + Embeddings)
# ------------------------------
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


def get_answer(question: str, context: str) -> str:
    """
    Generates an answer using Ollama with short-term and long-term memory.
    """
    try:
        # --- Retrieve from long-term memory (FAISS) ---
        docs = vectorstore.similarity_search(question, k=2)
        long_term_context = "\n".join([d.page_content for d in docs])

        # --- Get short-term memory ---
        chat_history_messages = short_term_memory.chat_memory.messages
        chat_history_text = ""
        if chat_history_messages:
            history_lines = []
            for msg in chat_history_messages[-6:]:  # Keep last few turns
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content}")
            chat_history_text = "\n".join(history_lines)

        # --- Create a smart prompt ---
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful and knowledgeable AI assistant.
            Use the **conversation history**, **long-term memory**, and **given context**
            to answer the question accurately and clearly, You should be answer based on the user query.
            If the information isn't available, respond honestly that you don't know.

            ### Conversation History:
            {chat_history}

            ### Long-Term Memory:
            {long_memory}

            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:
            """
        )

        formatted_prompt = prompt.format(
            chat_history=chat_history_text or "No previous conversation.",
            long_memory=long_term_context or "No relevant long-term memory.",
            context=context or "No external context provided.",
            question=question
        )

        # --- Query Ollama ---
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # --- Save in short-term memory ---
        short_term_memory.chat_memory.add_user_message(question)
        short_term_memory.chat_memory.add_ai_message(answer)

        # --- Add to long-term memory (FAISS) ---
        vectorstore.add_texts([f"Q: {question}\nContext: {context}\nA: {answer}"])
        vectorstore.save_local("faiss_index")

        return answer

    except Exception as e:
        import traceback
        print(f"Error in get_answer: {traceback.format_exc()}")
        return f"Error generating answer: {str(e)}"