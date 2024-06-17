import os
import streamlit as st

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY"

# Import necessary modules from LangChain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings  
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama

# Define the path to the data
DATA_PATH = "./rag_news_db"

# Initialize the vector store with Chroma, which is used for document retrieval
vectorstore = Chroma(persist_directory=DATA_PATH, embedding_function=OpenAIEmbeddings())
vectorstore.get()

# Pull the chat prompt template for retrieval-based QA from LangChain Hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Define the system message template for the chat prompt
system_template = """You are a stock advisor! Please answer the following question based on the data from the provided context, such as 10-K reports, financial news, or other relevant documents. Ensure your response is detailed, concise, and directly addresses the question. Use relevant financial metrics, trends, and insights from the context to support your answer.
"""

# Create the human message template for the chat prompt
human_prompt_content = """Answer the following question based on the provided context. You do not need to mention that the information is sourced from the context. Ensure your response includes summary from news, specific data points, financial metrics, and relevant insights:

<Context>
{context}
</Context>

Question: {input}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_prompt_content)
]

# Create the chat prompt template from the system and human message templates
prompt = ChatPromptTemplate.from_messages(messages)

# Initialize the language model with OpenAI's GPT-4 model
llm = ChatOpenAI(model="gpt-4o")

# Convert the vector store to a retriever object
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Create the document combination chain using the LLM and prompt
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)

# Create the retrieval chain using the retriever and document combination chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit UI for the stock advisor application
st.title("Stock Advisor")
question = st.text_input("Ask a question about Apple/Google/Nvidia/Tesla/Netflix")

# If a question is asked, use the retrieval chain to get the response
if question:
    response = retrieval_chain.invoke({"input": question})
    print(response)
    st.write(response["answer"])
