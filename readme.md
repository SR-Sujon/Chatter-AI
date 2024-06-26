# Chatter.AI
Chatter.AI is a web application that allows you to chat with websites and extract information from them. It uses the OpenAI API and LangChain library to process and understand the content of the website and allow users to interact with it on the Streamlit interface. It has two output formats: JSON and Q/A, including I/O configuration flexibility.

![Chatter AI Demo](https://github.com/SR-Sujon/Chatter-AI/blob/main/public/Chatter-ai-Cover.png)

## Deployed on Streamlit
Link: https://chatter-ai.streamlit.app/

## Prerequisites
To run Chatter.AI on your local machine, you need to have the followings installed:
* Anaconda / Miniconda
* Streamlit
* LangChain
* Python 3.10
* openai
* langchain-openai
* beautifulsoup4
* chromadb

In older version `.env` file was needed but now its not required anymore. Just directly paste your `OpenAI API key` in the interface Input.

## Installation
1. Install the dependencies by running the following command in the terminal:

```bash
conda create --name chatter-ai python=3.10
conda activate chatter-ai
pip install streamlit langchain langchain-openai beautifulsoup4 chromadb
```

2. Run the application by executing the following command in the terminal:

```bash
streamlit run src/app.py
```
or 

```bash
streamlit run chatter-ai.py
```

Both files are same. For streamlit deployment requirements, `chatter-ai.py` and `requirements.txt` files has been created. 

## Usage

1. Choose your preferred settings from the "I/O Configuration" section.
2. Enter your OPENAI API in the Input Terminal.
3. Enter the URL of the website you want to chat with in the "Website URL" input field.
4. Press "Enter" to connect with Chatter.AI.
5. Wait until Chatter.AI is available. It will greet you soon.
6. Ask Chatter.AI your relevant questions based on the website. It will provide you with detailed responses.
7. To reset sessions, click the "Reset session" button.

## Functions

* `check_output_format(OUTPUT_FORMAT)`: Returns a formatted string based on the output format.
* `get_vector_store_from_url(url)`: Creates a vector store from the URL in document format.
* `get_context_retreiver_chain(vector_store)`: Creates a retriever chain with the context from the vector store.
* `get_conversational_rag_chain(retriever_chain)`: Creates a conversational RAG chain with the retriever chain.
* `get_response(user_input)`: Returns a response from the conversational RAG chain.
* `reset_page()`: Reloads the page.

## Technologies Used
* Streamlit
* LangChain
* OpenAI
* ChromaDB
* Python


## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
