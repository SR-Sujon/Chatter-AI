# Setup conda environment and activate it.
# conda create --name chatter-ai python=3.10
# Import all required packages by the following command:
# pip install streamlit langchain openai

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

def get_response(user_input):
    return "I\'m sorry. I don\'t know."

# App configuration
st.set_page_config(page_title="Chatter.AI", page_icon="🧠", layout= "centered")
st.title('Chat with Websites')

# Schemas with session state object
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # AI giving Greetings
        AIMessage(content="Hello, I\'m Chatter AI. How can I help you?"),
    ]


# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url =="":
    st.info("Please enter a valid website URL")
else:
    # User Query
    user_query =  st.chat_input("Type your message here...") 

    if user_query is not None and user_query !="":
        response = get_response(user_query)
        # Append current queries to existing chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Debug conversations in sidebar
    #with st.sidebar:
    #    st.write(st.session_state.chat_history)

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)