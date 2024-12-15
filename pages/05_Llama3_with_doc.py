import streamlit as st
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(
    page_title="Document GPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    
    # 파일을 디스크에 저장
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # 캐시 디렉토리 경로 설정
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    
    # 캐시된 embeddings 파일이 존재하는지 확인
    if os.path.exists(f"./.cache/private_embeddings/{file.name}"):
        # 기존 embeddings 파일이 있으면 이를 로드
        embeddings = OllamaEmbeddings(model="llama3.1:latest")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.load_local(f"./.cache/private_embeddings/{file.name}", 
                                       cached_embeddings, allow_dangerous_deserialization=True)
    else:
        # 캐시된 embeddings 파일이 없으면 새로 생성
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OllamaEmbeddings(model="llama3.1:latest")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        vectorstore.save_local(f"./.cache/private_embeddings/{file.name}")
    
    retriever = vectorstore.as_retriever()
    return retriever
 

def save_messages(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_messages(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

#Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.

prompt = ChatPromptTemplate.from_template(
    """         
    Given the context, please provide a detailed and easy-to-understand explanation regarding the question. 
    Ensure to:
    1. Clarify the essence of the question, asking for additional details if necessary.
    2. Offer background information relevant to the question to set the stage for your explanation.
    3. Break down complex concepts or processes into manageable steps for clarity.
    4. Use examples and analogies to elucidate difficult concepts, making them relatable to everyday experiences.
    5. Summarize the main points and conclude your explanation, indicating areas for further exploration or where additional information might be beneficial.
    
    Aim to make your response comprehensive yet accessible, utilizing simple language to enhance understanding for all audience levels.

    Context: {context}
    Question: {question}
    """
)

st.title("Document GPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on sidebar
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
