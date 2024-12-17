import datetime

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

st.set_page_config(
    page_title="Search Groq",
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
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-70b-8192",
]

with st.sidebar:
    selected_option = st.selectbox("Select a model:", options, index=0)
    max_results = st.slider(
        "Number of search results",
        min_value=1,
        max_value=10,
        value=5,
        help="Choose how many documents you want to receive as search results",
    )

if "search_messages" not in st.session_state:
    st.session_state["search_messages"] = []

llm = ChatGroq(
    temperature=0.1,
    model_name=selected_option,
    streaming=False,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
)

tool = TavilySearchResults(
    max_results=max_results,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

if "search_chat_summary" not in st.session_state:
    st.session_state["search_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["search_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["search_messages"].append(
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
    if "search_messages" in st.session_state:
        for message in st.session_state["search_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])


today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are a helpful assistant that provides detailed answers based on search results.
            Please provide search results based on the latest possible date.
            Today's date is: {today}
            """,
        ),
        (
            "human",
            """Please answer the following question based on the search results:
    
Search Results: {search_results}
Question: {question}

Please format your answer as follows:
1. Main Summary
2. Detailed Information
3. Sources Used""",
        ),
    ]
)


def save_context(question, result):
    st.session_state["search_chat_summary"].append(
        {
            "question": question,
            "answer": result,
        }
    )


def invoke_chain(question):
    try:
        # llm_chain 사용하여 직접 실행
        result = llm_chain.invoke(question)
        save_context(question, result.content)
        return result.content

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Sorry. An error occurred while generating the response."


# 새로운 체인 정의
llm_chain = (
    {
        "question": RunnablePassthrough(),
        "search_results": lambda x: tool.invoke({"query": x}),
    }
    | prompt
    | llm
)


st.title("🔍 Search Groq")

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

    # 대화 기록 표시
    if "search_messages" in st.session_state:
        paint_history()
    else:
        send_message("I'm ready! Ask away!", "ai", save=True)

    authentication_status = st.session_state["authentication_status"]
    if authentication_status:
        message = st.chat_input("Ask anything about something...")
        if message:
            send_message(message, "human")

            with st.chat_message("ai"):
                with st.spinner("Generating answers..."):
                    try:
                        response = invoke_chain(message)
                        st.markdown(response)
                        save_messages(response, "ai")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    else:
        st.markdown(
            "<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>",
            unsafe_allow_html=True,
        )
