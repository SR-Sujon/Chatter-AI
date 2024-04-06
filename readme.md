# Chatter.AI
Chatter.AI is a web application that allows you to chat with websites and extract information from them. It uses the OpenAI API and LangChain library to process and understand the content of the website.

![Chatter AI Demo](https://github.com/SR-Sujon/Chatter-AI/blob/main/public/Chatter-ai-Cover.jpg)

## Prequisites
To run Chatter.AI, you need to have the following installed:
* Anaconda / Miniconda
* Streamlit
* Python 3.10

You also need to create a `.env` file in the root directory of the project and add your OpenAI API key to it:

```bash
OPEN_AI_API_KEY=your_openai_api_key
```
## Installation
1. Install the dependencies by running the following command in the terminal:

```bash
conda create --name chatter-ai python=3.10
conda activate chatter-ai
pip install -r requirements.txt
```

2. Run the application by executing the following command in the terminal:

```bash
streamlit run src\app.py
```

## Usage

1. Choose your preferred settings from the "I/O Configuration" section.
2. Enter the URL of the website you want to chat with in the "Website URL" input field.
3. Press "Enter" to connect with Chatter.AI.
4. Wait until Chatter.AI is available. It will greet you soon.
5. Ask Chatter.AI your relevant questions based on the website. It will provide you with detailed responses.

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