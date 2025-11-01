import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ------------------------------
# Global LLM (Groq)
# ------------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# ------------------------------
# Short-term memory (per session)
# ------------------------------
short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


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
    Generates an answer with short-term and long-term memory.
    """
    try:
        # Retrieve relevant long-term memory from FAISS
        docs = vectorstore.similarity_search(question, k=2)
        long_term_context = "\n".join([d.page_content for d in docs])
        
        # Get chat history
        chat_history_messages = short_term_memory.chat_memory.messages
        chat_history_text = ""
        if chat_history_messages:
            history_lines = []
            for msg in chat_history_messages[-6:]:  # Last 3 exchanges
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content}")
            chat_history_text = "\n".join(history_lines)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Use the **conversation history**, 
**long-term memory**, and **provided context** to answer the question. 
If the answer cannot be found, say so clearly.

### Conversation History:
{chat_history}

### Long-Term Memory:
{long_memory}

### Context:
{context}

### Question:
{question}

### Answer:"""
        )
        
        # Format prompt with all context
        formatted_prompt = prompt.format(
            chat_history=chat_history_text or "No previous conversation",
            long_memory=long_term_context,
            context=context,
            question=question
        )
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Save to short-term memory
        short_term_memory.chat_memory.add_user_message(question)
        short_term_memory.chat_memory.add_ai_message(answer)
        
        # Store Q&A into long-term memory
        vectorstore.add_texts([f"Q: {question}\nContext: {context}\nA: {answer}"])
        vectorstore.save_local("faiss_index")

        return answer

    except Exception as e:
        import traceback
        print(f"Error in get_answer: {traceback.format_exc()}")
        return f"Error generating answer: {str(e)}"
    

