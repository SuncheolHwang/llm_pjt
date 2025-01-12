import streamlit as st
import streamlit_authenticator as stauth

# import yaml
# from yaml.loader import SafeLoader

st.set_page_config(
    page_title="LLM Collection for Office",
    page_icon="💀",
)

# with open("./config.ymal") as file:
#     config = yaml.load(file, Loader=SafeLoader)

username = list(st.secrets["credentials"]["usernames"].keys())[0]

credentials = {
    "credentials": {
        "usernames": {
            username: {
                "email": st.secrets["credentials"]["usernames"][username]["email"],
                "name": st.secrets["credentials"]["usernames"][username]["name"],
                "password": st.secrets["credentials"]["usernames"][username][
                    "password"
                ],
            }
        }
    },
    "cookie": {
        "expiry_days": st.secrets["cookie"]["expiry_days"],
        "key": st.secrets["cookie"]["key"],
        "name": st.secrets["cookie"]["name"],
    },
    "preauthorized": {
        "emails": st.secrets["preauthorized"]["emails"],
    },
}

authenticator = stauth.Authenticate(
    # config["credentials"],
    # config["cookie"]["name"],
    # config["cookie"]["key"],
    # config["cookie"]["expiry_days"],
    # config["preauthorized"],
    credentials["credentials"],
    credentials["cookie"]["name"],
    credentials["cookie"]["key"],
    credentials["cookie"]["expiry_days"],
    credentials["preauthorized"],
)

name, authentication_status, username = authenticator.login()

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = authentication_status

if authentication_status:
    authenticator.logout("Logout", "main")
    st.write(f"Welcome *{name}*")
    st.markdown(
        """
# 1. Groq Llama3
    """
    )
elif authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
