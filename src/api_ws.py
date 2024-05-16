from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
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
import json

from src.schemas import ChatRequest, ChatResponse

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7)
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


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7)
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


# Load html
with open("src/index.html", "r") as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    vector_store = None

    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            url = data_json["url"]
            message = data_json["message"]

            if vector_store is None:
                vector_store = get_vectorstore_from_url(url)

            response_text = get_response(vector_store, message, chat_history)
            chat_history.append(HumanMessage(content=message))
            chat_history.append(AIMessage(content=response_text))

            await websocket.send_text(response_text)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
