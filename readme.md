# Stock News and Financial Reports Retrieval and Embedding (RAG)

This project retrieves stock news and financial reports, processes the content, and embeds it into a vector store for later retrieval using a chatbot interface.

## Project Structure

- **main.py**: The main script to run the Streamlit application.
- **utils.py**: Contains utility functions for URL sanitization, content fetching, and PDF processing.
- **data/**: Directory to store the news database and PDF files.

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- Pandas
- Requests
- BeautifulSoup4
- Feedparser
- StockNews

## Installation


1. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

2. Set your OpenAI API key as an environment variable:

    ```sh
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

## Usage

### Running the Application

1. Fetch news for specific stocks and embed the content:

    ```sh
    python data_processing.py
    ```

2. Run the Streamlit application:

    ```sh
    streamlit run main.py
    ```

3. Open your browser and navigate to `http://localhost:8501` to use the application.

### Code Overview

#### main.py

This script sets up the Streamlit interface for the stock advisor application.

- **Imports**: Necessary modules from LangChain and other libraries.
- **OpenAI API Key**: Sets the OpenAI API key from environment variables.
- **Constants**: Defines paths for data storage.
- **Streamlit UI**: Creates a simple UI for asking stock-related questions.
- **Retrieval Chain**: Initializes the retrieval chain for fetching and processing stock news and reports.

#### data_processing.py

This script contains utility functions for fetching and processing news and PDF content.

- **sanitize_url(url)**: Sanitizes a URL by removing query parameters.
- **fetch_content(url)**: Fetches and parses the content of a webpage.
- **fetch_news_for_stocks(stocks, news_file='simple_news.csv')**: Fetches news for given stock symbols and saves it to a CSV file.
- **get_stock_name(symbol)**: Converts a stock symbol to a stock name.
- **get_text_chunks_langchain(text, stock)**: Splits text into chunks and creates Document objects for each chunk.
- **load_news(news_file)**: Loads news data from a CSV file.
- **embed_news(news_file)**: Embeds news content and saves the vectors.
- **load_pdfs_from_directory(directory)**: Loads and processes PDFs from a directory.

### Example Usage

1. Define a list of stock symbols and names:

    ```python
    stocks = ['AAPL', 'GOOGL', 'TSLA', 'NFLX', 'NVDA']
    stock_names = ['Apple', 'Google', 'Tesla', 'Netflix', 'NVIDIA']
    ```

2. Fetch and embed news for these stocks:

    ```python
    fetch_news_for_stocks(stocks, news_file='simple_news.csv')
    embed_news('simple_news.csv')
    ```

3. Load and process PDFs from a directory:

    ```python
    docs = load_pdfs_from_directory(PDF_FOLDER)
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=DATA_PATH)
    vectorstore.persist()
    ```

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the powerful language model chaining tools.
- [OpenAI](https://openai.com/) for providing the language models.
- [Streamlit](https://www.streamlit.io/) for the easy-to-use web application framework.
- [Yahoo Finance](https://finance.yahoo.com/) for the stock news feeds.
