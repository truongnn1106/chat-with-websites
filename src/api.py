from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define the FastAPI app
app = FastAPI()


# Define the request and response models
class ChatRequest(BaseModel):
    url: str = "https://vucar.vn/"
    message: str
    chat_history: list = []


class ChatResponse(BaseModel):
    answer: str
    chat_history: list


# Define utility functions
def get_vectorstore_from_url(url: str) -> Chroma:
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store: Chroma):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=1000)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain: callable):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=1000)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(vector_store, user_input, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke(
        {"chat_history": chat_history, "input": user_input}
    )
    return response["answer"]


# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        vector_store = get_vectorstore_from_url(request.url)
        response_text = get_response(
            vector_store, request.message, request.chat_history
        )
        request.chat_history.append(HumanMessage(content=request.message))
        request.chat_history.append(AIMessage(content=response_text))
        return ChatResponse(answer=response_text, chat_history=request.chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
