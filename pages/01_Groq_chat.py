import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Groq",
    page_icon="📃",
)

callback = False


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    thinking = ""
    in_think_block = False

    def on_llm_start(self, *args, **kwargs):
        if callback:
            self.message_box = st.empty()
            self.message = ""
            self.thinking = ""
            self.in_think_block = False

    def on_llm_end(self, *args, **kwargs):
        if callback:
            # 최종 메시지를 표시
            final_message = ""
            if self.thinking:
                final_message += f'<div class="think-box">{self.thinking}</div>'
            final_message += self.message
            save_messages(final_message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        if callback:
            if token.startswith("<think>"):
                self.in_think_block = True
            elif token.startswith("</think>"):
                self.in_think_block = False
            else:
                if self.in_think_block:
                    self.thinking += token
                else:
                    self.message += token

            # 현재까지의 전체 메시지를 표시
            display_message = ""
            if self.thinking:
                display_message += f'<div class="think-box">{self.thinking}</div>'
            display_message += self.message
            self.message_box.markdown(display_message, unsafe_allow_html=True)


options = [
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-70b-8192",
]

with st.sidebar:
    selected_option = st.selectbox("Select a model:", options, index=0)

    prompt_text = st.text_area(
        "Prompt",
        """
Description: As a hardware and software expert, your job is to provide detailed and easy-to-understand explanations in Korean for inquiries or statements. 
You need to explain complex concepts related to hardware and software in an easy-to-understand manner so that they can be understood by a wide audience. 
Your goal is not only to simplify the explanation, but also to provide relevant examples wherever possible to enhance the questioner's understanding.

    Role: Hardware and software expert
    Goal: Help the questioner gain a comprehensive understanding of hardware and software concepts.

    Guidelines
    1. break down complex technical topics into simple, easy-to-understand explanations.
    2. Use clear, accessible language that individuals without a technical background can understand.
    3. Provide real-world examples or hypothetical scenarios to illustrate explanations and make abstract concepts concrete.
    4. Explain the "how" and "why" of processes and technologies to deepen the questioner's understanding.
    5. When discussing software, explain how the software interacts with hardware to perform its function.
""".strip(),
        #         """설명: 하드웨어 및 소프트웨어 전문가로서 당신의 임무는 문의나 진술에 대해 상세하고 이해하기 쉬운 설명을 제공하는 것입니다.
        # 하드웨어 및 소프트웨어와 관련된 복잡한 개념을 이해하기 쉽게 설명하여 광범위한 청중이 이해할 수 있도록 해야 합니다.
        # 설명을 단순화할 뿐만 아니라 가능한 한 관련 사례를 제시하여 질문자의 이해도를 높이는 것이 목표입니다.
        # 역할: 하드웨어 및 소프트웨어 전문가
        # 목표: 질문자가 하드웨어 및 소프트웨어 개념을 포괄적으로 이해할 수 있도록 돕는다.
        # 가이드라인
        # 1. 복잡한 기술 주제를 간단하고 이해하기 쉬운 설명으로 세분화합니다.
        # 2. 기술적 배경 지식이 없는 개인도 이해할 수 있는 명확하고 접근하기 쉬운 언어를 사용합니다.
        # 3. 실제 사례나 가상의 시나리오를 제시하여 설명의 실례를 보여주고 추상적인 개념을 구체화합니다.
        # 4. 프로세스와 기술의 '방법'과 '이유'를 설명하여 질문자의 이해를 깊게 합니다.
        # 5. 소프트웨어에 대해 논의할 때는 소프트웨어가 하드웨어와 어떻게 상호작용하여 기능을 수행하는지 설명하세요.
        # 주제 예시
        # - CPU는 어떻게 명령을 처리하나요?
        # - 컴퓨터에서 운영 체제의 역할은 무엇인가요?
        # - 스마트폰이 하드웨어와 소프트웨어를 모두 사용하여 사진을 캡처하고 처리하는 방법을 설명할 수 있나요?
        # - 코드를 실행 가능한 프로그램으로 컴파일하는 과정을 설명할 수 있나요?
        # """,
        # """Explanation: As a hardware and software expert, your task is to provide detailed and easily understandable explanations in response to inquiries or statements.
        # You are expected to demystify complex concepts related to both hardware and software, making them accessible to a broad audience.
        # Your objective is to enhance the questioner's comprehension by not only simplifying explanations but also by providing relevant examples whenever possible.
        # Role: Hardware and Software Expert
        # Objective: To assist questioners in comprehensively understanding hardware and software concepts
        # Guidelines:
        # 1. Break down complex technical topics into simple, digestible explanations.
        # 2. Use clear and accessible language that can be understood by individuals without a technical background.
        # 3. Provide real-life examples or hypothetical scenarios to illustrate your explanations and make abstract concepts tangible.
        # 4. Address the 'how' and 'why' behind processes and technologies to deepen the questioner's understanding.
        # 5. When discussing software, explain how it interacts with hardware to perform its functions.
        # Example Topics:
        # - How do CPUs process instructions?
        # - What is the role of an operating system in a computer?
        # - Can you explain how a smartphone uses both hardware and software to capture and process photos?
        # - Describe the process of compiling code into an executable program.
        # When answering, it's important to remember that your goal is to make the information as accessible as possible.
        # Strive to not only answer the question but also to educate the questioner, providing them with a foundation that enables them to grasp more complex concepts in the future."""
    )

if "groq_messages" not in st.session_state:
    st.session_state["groq_messages"] = []

llm = ChatGroq(
    temperature=0.1,
    model_name=selected_option,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
)

if "groq_chat_summary" not in st.session_state:
    st.session_state["groq_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["groq_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["groq_messages"].append(
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
    for message in st.session_state["groq_messages"]:
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
    st.session_state["groq_chat_summary"].append(
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


st.title("Groq-Llama3 Chatbot")

st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
    }
    .think-box {
        background-color: rgba(240, 242, 246, 0.1);  /* 더 어두운 배경색 */
        border-left: 3px solid #7c8494;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-style: italic;
        color: inherit;  /* 현재 테마의 텍스트 색상 사용 */
    }
    /* 다크 모드 대응 */
    [data-theme="dark"] .think-box {
        background-color: rgba(49, 51, 63, 0.8);
        border-left: 3px solid #4c5156;
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
