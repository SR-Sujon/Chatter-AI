# ENVIRONMENT CONFIG:
# -----------------------------------------------------------------
# install anaconda / miniconda
# Setup conda environment and activate it by the following command from vscode terminal:
# conda create --name chatter-ai python=3.10
# conda activate chatter-ai

# DEPENDENCIES:
# -----------------------------------------------------------------
# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb

# STEPS: 
# -----------------------------------------------------------------
# 1. Install dependencies (if not already installed)
# 2. Follow instructions on landing page to provide perfect prompts 
# 3. Chatter AI then extracts text from HTML files, split it into different chunks of text in doc format
# 4. Then uses OpenAIEmbeddings to create a vector store with Chroma
# 5. After that creates a Retreiver Chain by embedding the entire conversation history to retreive relevant chunks of information
# 6. Finally, creates a Conversation RAG Chain with system input and user input appended with the prompt 


# IMPORT LIBRARIES
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load OPEN_AI_API KEY from the environment file
load_dotenv()


# CHECK OUTPUT FORMAT
def check_output_format(OUTPUT_FORMAT):
    if OUTPUT_FORMAT == "JSON":
        return "Extract information from the document and consider the [SCHEMAS] as JSON elements and rewrite them in a JSON format efficiently ensuring completeness and accuracy for optimal utilization."
    elif OUTPUT_FORMAT == 'Q/A':
        return "Utilize the extracted information from the document to craft precise responses to predefined questions. Ensure responses remain succinct, containing no more than 100 words."
        

# perform vector embeddings on the received data
def get_vector_store_from_url(url):
    # get the vector store from the url in document format
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the doc into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the retreived chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store


# gather context information about the conversation history and adds user prompt
def get_context_retreiver_chain(vector_store):
    # init language model
    llm = ChatOpenAI(temperature=TEMPARATURE, model=MODEL)
    # allows retrieve relevant text from vector_store
    retriever = vector_store.as_retriever()
    # init prompt, populated with chat_history
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user", SEARCH_QUERY_PROMPT)
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# get information relevant to the conversation including user queries    
def get_conversational_rag_chain(retriever_chain):
    # initialize llm
    llm = ChatOpenAI(temperature=TEMPARATURE, model=MODEL)
    # initialize prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
    ])
    
    # Pass prams- llm, prompt
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    # Plugging two chains together
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

    
# generate a retriever chain and conversational_rag_chain for each user query, then return a response with chat_history and user input.
def get_response(user_input):
    # Fetch retriver chain context through session state vector_store
    retriever_chain = get_context_retreiver_chain(st.session_state.vector_store)
    
    
    # Get the conversation rag chain
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    
    response = conversational_rag_chain.invoke({
            "chat_history":st.session_state.chat_history,
            "input": user_input +". "+ SEARCH_QUERY_PROMPT
        })
    
    return response['answer']
    
# Reset session   
def reset_page():
    st.markdown("<script>location.reload(true);</script>", unsafe_allow_html=True)    


# app configuration
st.set_page_config(page_title="Chatter.AI", page_icon=":snowflake:", layout= "centered")
st.image("src/brainy.png")
st.title('Chatter.AI - Chat with Websites')


# Sidebar
with st.sidebar:
    st.header("Settings")
    st.info("Before entering the website link, please choose your preferred settings from the list from I/O configuaration below.")
    website_url = st.text_input("Website URL")
     # Create a button to clear the input field
    if st.button("Reset Session"):
        website_url = ""
        reset_page()
    
    st.subheader("I/O Configuration")
    OUTPUT_FORMAT = st.selectbox("Output format:",["JSON","Q/A"])
    SEARCH_QUERY_PROMPT = check_output_format(OUTPUT_FORMAT)
    
    MODEL = st.selectbox("Model:",["gpt-3.5-turbo-0125","gpt-3.5-turbo-0163","gpt-3.5-turbo-1106"])
    
    TEMPARATURE = st.slider("Temparature:",0.0, 1.0)
    
    
        
# Main panel
if website_url is None or website_url =="":
    st.subheader("Instructions")
    st.info("FOLLOW THE INSTRUCTIONS PROPERLY: \n\n * Please select all required I/O configuration. \n\n * Then enter a valid website URL (ex: https://en.wikipedia.org/wiki/OpenAI)\n\n * After filling the website url section, press 'Enter' to connect with Chatter AI. \n\n * Wait until Chatter AI is available. It will greet you soon.\n\n * After that, you can ask Chatter AI your relevant questions based on that website, It will provide you with detailed response.\n\n ALL THE BEST! \n\n Powered by OpenAI and LangChain. \n\n Developed by Saidur Rahman Sujon. \n\n")
    
else:
    # Session state: Initialize chat_history at the beginning of each session
    st.success("Session status: Connected to server")
    if "chat_history" not in st.session_state:
        if OUTPUT_FORMAT == "Q/A":
            st.session_state.chat_history = [
                # AI giving Greetings
                AIMessage(content="Hello, I\'m Chatter AI. How can I help you?"),
            ]
        elif OUTPUT_FORMAT == "JSON":
            st.info("Enter the [SCHEMAS] in your prompt in the following format: \n\n SCHEMAS: product title, product price etc.")
            st.session_state.chat_history = []    
        
    # Preventing re-embedding every single time making a query    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)
    

    
    # Users Input
    user_query =  st.chat_input("Type your message here...") 

    if user_query is not None and user_query !="":
        response = get_response(user_query)
        
        
        if OUTPUT_FORMAT == 'JSON':
            st.json(response)
            
        elif OUTPUT_FORMAT == 'Q/A':
            # Append current queries to existing chat history
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        
        # Testing if retreiver chain is working properly
        #---------------------------------------------------
        #retrieved_documents = retriever_chain.invoke({
        #    "chat_history": st.session_state.chat_history,
        #    "input":user_query
        #})
        #st.write(retrieved_documents)
        

    # Display Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)