import streamlit as st
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import os
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain.memory import ConversationTokenBufferMemory

set_llm_cache(SQLiteCache("cache.db"))

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(
    page_title="Document GPT(Phi3)",
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

if "Phi3_doc_messages" not in st.session_state:
    st.session_state["Phi3_doc_messages"] = []

llm = ChatOllama(
    model="phi3:3.8b",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
)

if "Phi3_doc_summary" not in st.session_state:
    st.session_state["Phi3_doc_summary"] = []

    for chat_list in st.session_state["Phi3_doc_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model="phi3:3.8b")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver
 

def save_messages(message, role):
    st.session_state["Phi3_doc_messages"].append(
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
    for message in st.session_state["Phi3_doc_messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(_):
    loaded_memory = memory.load_memory_variables({})["history"]
    return loaded_memory

def save_context(question, result):
    st.session_state["Phi3_doc_summary"].append(
        {
            "question": question,
            "answer": result,
        }
    )

def invoke_chain(question):
    result = chain.invoke(
        {"question": question},
    )
    save_context(message, result.content)

# prompt = ChatPromptTemplate.from_template(
#     """         
#     Given the context, please provide a detailed and easy-to-understand explanation regarding the question. 
#     Ensure to:
#     1. Clarify the essence of the question, asking for additional details if necessary.
#     2. Offer background information relevant to the question to set the stage for your explanation.
#     3. Break down complex concepts or processes into manageable steps for clarity.
#     4. Use examples and analogies to elucidate difficult concepts, making them relatable to everyday experiences.
#     5. Summarize the main points and conclude your explanation, indicating areas for further exploration or where additional information might be beneficial.
#     6. Make your answer contextually relevant to the previous conversation(history).

#     Aim to make your response comprehensive yet accessible, utilizing simple language to enhance understanding for all audience levels.

#     Context: {context}
#     History: {history}
#     Question: {question}
#     """
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the context, please provide a detailed and easy-to-understand explanation regarding the question. 
            Ensure to:
            1. Clarify the essence of the question, asking for additional details if necessary.
            2. Offer background information relevant to the question to set the stage for your explanation.
            3. Break down complex concepts or processes into manageable steps for clarity.
            4. Use examples and analogies to elucidate difficult concepts, making them relatable to everyday experiences.
            5. Summarize the main points and conclude your explanation, indicating areas for further exploration or where additional information might be beneficial.
            6. Make your answer contextually relevant to the previous conversation(history).

            Aim to make your response comprehensive yet accessible, utilizing simple language to enhance understanding for all audience levels.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

st.title("Document GPT(Phi3)")

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
                "history": RunnableLambda(load_memory),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            invoke_chain(message)
else:
    st.session_state["messages"] = []
