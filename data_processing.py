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
from langchain.embeddings import GPT4AllEmbeddings, OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PyPDFDirectoryLoader, UnstructuredPDFLoader, UnstructuredHTMLLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from stocknews import StockNews
import feedparser
import pandas as pd
import datetime as dt
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse

# Constants for data and PDF folder paths
DATA_PATH = "./rag_news_db"
PDF_FOLDER = "./pdf"

def sanitize_url(url):
    """
    Sanitize a URL by removing query parameters.
    """
    parsed_url = urlparse(url)
    sanitized_url = urlunparse(parsed_url._replace(query=""))
    return sanitized_url

def fetch_content(url):
    """
    Fetch and parse the content of a webpage.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def fetch_news_for_stocks(stocks, news_file='simple_news.csv'):
    """
    Fetch news for given stock symbols and save to a CSV file.
    """
    YAHOO_URL = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'
    
    # Create or load the DataFrame
    if os.path.exists(news_file):
        df = pd.read_csv(news_file)
    else:
        df = pd.DataFrame(columns=['guid', 'stock', 'title', 'summary', 'published', 'link', 'content'])

    # Fetch and process news for each stock
    new_entries = []
    for stock in stocks:
        feed = feedparser.parse(YAHOO_URL % stock)
        for entry in feed.entries:
            if entry.guid in df['guid'].values:
                continue

            published_date = dt.datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S +0000')
            sanitized_link = sanitize_url(entry.link)
            content = fetch_content(sanitized_link)

            new_entries.append({
                'guid': entry.guid,
                'stock': stock,
                'title': entry.title,
                'summary': entry.summary,
                'published': published_date,
                'link': entry.link,
                'content': content
            })

    # Convert new entries to DataFrame and concatenate with existing DataFrame
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        df = pd.concat([df, new_df], ignore_index=True)

    # Save the DataFrame to CSV
    df.to_csv(news_file, index=False)

# Example usage
stocks = ['AAPL', 'GOOGL', 'TSLA', 'NFLX', 'NVDA']
stock_names = ['Apple', 'Google', 'Tesla', 'Netflix', 'NVIDIA']

# Create a dictionary mapping symbols to names
symbol_to_name = dict(zip(stocks, stock_names))

def get_stock_name(symbol):
    """
    Convert stock symbol to stock name.
    """
    return symbol_to_name.get(symbol, "Unknown")

def get_text_chunks_langchain(text, stock):
    """
    Split text into chunks and create Document objects for each chunk.
    """
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    stock_name = get_stock_name(stock)
    docs = [Document(page_content=f"{stock_name}_news: {x}") for x in text_splitter.split_text(text)]
    return docs

def load_news(news_file):
    """
    Load news data from a CSV file.
    """
    if os.path.exists(news_file):
        df = pd.read_csv(news_file)
        return df

def embed_news(news_file):
    """
    Embed news content and save the vectors.
    """
    news = load_news(news_file)

    # Loop through the DataFrame and process the content
    for index, row in news.iterrows():
        stock = str(row['stock'])  # Ensure stock is treated as a string
        content = str(row['content'])  # Ensure content is treated as a string
        docs = get_text_chunks_langchain(content, stock)
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=DATA_PATH)
        vectorstore.persist()
        
# Fetch news for stocks and embed the news content
fetch_news_for_stocks(stocks, news_file='simple_news.csv')
embed_news('simple_news.csv')

def load_pdfs_from_directory(directory):
    """
    Load and process PDFs from a directory.
    """
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)  # Adjust chunk size and overlap as needed

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            stock_name = filename.rsplit(".", 1)[0]  # Extract stock name from filename
            file_path = os.path.join(directory, filename)
            loader = UnstructuredPDFLoader(file_path)
            pdf_content = loader.load()

            # Convert PDF content to string
            text = "".join([page.page_content for page in pdf_content])
            
            # Split the text into chunks
            chunks = text_splitter.split_text(text)
            docs.extend([Document(page_content=f"{stock_name}_10K_Report: {chunk}") for chunk in chunks])

    return docs

# Load PDFs from the specified directory, embed the content, and save the vectors
docs = load_pdfs_from_directory(PDF_FOLDER)
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=DATA_PATH)      
vectorstore.persist()
