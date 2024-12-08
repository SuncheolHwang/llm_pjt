# import datetime

# import streamlit as st
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.memory import ConversationTokenBufferMemory
# from langchain_groq import ChatGroq
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.runnables import RunnableConfig, chain

# st.set_page_config(
#     page_title="Search GPT",
#     page_icon="📃",
# )

# callback = False


# class ChatCallbackHandler(BaseCallbackHandler):
#     message = ""

#     def on_llm_start(self, *args, **kwargs):
#         if callback:
#             self.message_box = st.empty()

#     def on_llm_end(self, *args, **kwargs):
#         if callback:
#             save_messages(self.message, "ai")

#     def on_llm_new_token(self, token, *args, **kwargs):
#         if callback:
#             self.message += token
#             self.message_box.markdown(self.message)


# options = [
#     "llama-3.3-70b-versatile",
#     "llama-3.1-70b-versatile",
#     "llama3-groq-70b-8192-tool-use-preview",
#     "llama3-70b-8192",
# ]

# with st.sidebar:
#     selected_option = st.selectbox("Select a model:", options, index=0)

#     prompt_text = st.text_area(
#         "Prompt",
#         """
#         Summary all contents.
#         """,
#     )

# if "groq_messages" not in st.session_state:
#     st.session_state["groq_messages"] = []

# llm = ChatGroq(
#     temperature=0.1,
#     model_name=selected_option,
#     streaming=True,
#     callbacks=[ChatCallbackHandler()],
# )

# memory = ConversationTokenBufferMemory(
#     llm=llm,
#     max_token_limit=1000,
#     return_messages=True,
# )

# tool = TavilySearchResults(
#     max_results=5,
#     search_depth="advanced",
#     include_answer=True,
#     include_raw_content=True,
#     include_images=False,
#     # include_domains=[...],
#     # exclude_domains=[...],
#     # name="...",            # overwrite default tool name
#     # description="...",     # overwrite default tool description
#     # args_schema=...,       # overwrite default args_schema: BaseModel
# )

# if "groq_chat_summary" not in st.session_state:
#     st.session_state["groq_chat_summary"] = []
# else:
#     callback = False
#     for chat_list in st.session_state["groq_chat_summary"]:
#         memory.save_context(
#             {"input": chat_list["question"]},
#             {"output": chat_list["answer"]},
#         )


# def save_messages(message, role):
#     st.session_state["groq_messages"].append(
#         {
#             "message": message,
#             "role": role,
#         }
#     )


# def send_message(message, role, save=True):
#     with st.chat_message(role):
#         st.markdown(message)
#     if save:
#         save_messages(message, role)


# def paint_history():
#     for message in st.session_state["groq_messages"]:
#         send_message(message["message"], message["role"], save=False)


# today = datetime.datetime.today().strftime("%D")
# prompt = ChatPromptTemplate(
#     [
#         ("system", f"You are a helpful assistant. The date today is {today}."),
#         ("human", "{user_input}"),
#         ("placeholder", "{messages}"),
#     ]
# )


# def load_memory(_):
#     return memory.load_memory_variables({})["history"]


# def save_context(question, result):
#     st.session_state["groq_chat_summary"].append(
#         {
#             "question": question,
#             "answer": result,
#         }
#     )


# # specifying tool_choice will force the model to call this tool.
# llm_with_tools = llm.bind_tools([tool])

# llm_chain = prompt | llm_with_tools


# @chain
# def tool_chain(user_input: str, config: RunnableConfig):
#     input_ = {"user_input": user_input}
#     ai_msg = llm_chain.invoke(input_, config=config)
#     tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
#     return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


# def invoke_chain(question):
#     result = tool_chain.invoke(
#         {"question": question},
#     )
#     save_context(message, result.content)


# st.title("Groq-Llama3 Chatbot")

# st.markdown(
#     """
#     <style>
#     .big-font {
#         font-size:30px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# if "authentication_status" not in st.session_state:
#     st.markdown(
#         "<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>",
#         unsafe_allow_html=True,
#     )
# else:
#     st.markdown(
#         """
#         Welcome!

#         Use this chatbot to ask questions to an AI about your Questions!

#         """
#     )

#     send_message("I'm ready! Ask away!", "ai", save=False)
#     paint_history()

#     authentication_status = st.session_state["authentication_status"]
#     if authentication_status:
#         message = st.chat_input("Ask anything about something...")
#         if message:
#             send_message(message, "human")
#             # chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

#             with st.chat_message("ai"):
#                 callback = True
#                 invoke_chain(message)
#     else:
#         st.markdown(
#             "<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>",
#             unsafe_allow_html=True,
#         )
