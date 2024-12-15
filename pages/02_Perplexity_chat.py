import streamlit as st
from langchain_community.chat_models import ChatPerplexity
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory

st.set_page_config(
    page_title="Perplexity",
    page_icon="🔍",
)

callback = False


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        if callback:
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        if callback:
            save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        if callback:
            self.message += token
            self.message_box.markdown(self.message)


options = [
    "llama-3.1-sonar-small-128k-online",
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-huge-128k-online",
]

with st.sidebar:
    selected_option = st.selectbox("Select a model:", options, index=0)

    prompt_text = st.text_area(
        "Prompt",
        "You are a helpful assistant.",
    )

if "perplexity_messages" not in st.session_state:
    st.session_state["perplexity_messages"] = []

llm = ChatPerplexity(
    temperature=0.1,
    model=selected_option,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
)

if "perplexity_chat_summary" not in st.session_state:
    st.session_state["perplexity_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["perplexity_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["perplexity_messages"].append(
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
    for message in st.session_state["perplexity_messages"]:
        send_message(message["message"], message["role"], save=False)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {prompt_text}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def save_context(question, result):
    st.session_state["perplexity_chat_summary"].append(
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


st.title("Perplexity Chatbot")

st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "authentication_status" not in st.session_state:
    st.markdown(
        "<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        Welcome!
        
        Use this chatbot to ask questions to an AI about your Questions!
        
        """
    )

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    authentication_status = st.session_state["authentication_status"]
    if authentication_status:
        message = st.chat_input("Ask anything about something...")
        if message:
            send_message(message, "human")
            chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

            with st.chat_message("ai"):
                callback = True
                invoke_chain(message)
    else:
        st.markdown(
            "<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>",
            unsafe_allow_html=True,
        )
