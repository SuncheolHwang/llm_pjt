import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory

st.set_page_config(
    page_title="Open AI Chat",
    page_icon="🤖",
)

callback = False

# GPT 모델 옵션 추가
gpt_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o1",
]

with st.sidebar:
    selected_model = st.selectbox("Select GPT Model:", gpt_models, index=0)
    prompt_text = st.text_area(
        "Prompt",
        """You are an engineering expert. Answer the question briefly and to the point.""",
    )


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


if "gpt4_messages" not in st.session_state:
    st.session_state["gpt4_messages"] = []

# 선택된 모델을 llm 초기화에 사용
llm = ChatOpenAI(
    temperature=0.1,
    model=selected_model,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationBufferWindowMemory(
    llm=llm,
    k=1,
    return_messages=True,
)

if "gpt4_chat_summary" not in st.session_state:
    st.session_state["gpt4_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["gpt4_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["gpt4_messages"].append(
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
    for message in st.session_state["gpt4_messages"]:
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
    st.session_state["gpt4_chat_summary"].append(
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


st.title("Open AI Chatbot")

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
