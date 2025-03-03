# Web_task


Home Route (/):

Displays a landing page (index.html).
Scrape Data (/load):

Accepts a URL from the user.
Scrapes the URL for various textual elements (headers, paragraphs, etc.).
Embeds the scraped text using a Sentence-BERT model.
Inserts the embedded text data into Milvus.
Query Data (/query):

Accepts a user query (question).
Embeds the query using the same Sentence-BERT model.
Searches the Milvus database for the most similar documents.
Uses a large language model (Mixtral) to generate an answer based on the retrieved context.
Renders the results in an HTML template (index2.html).
