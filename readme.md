<h1>Kilimo AI</h1>

<p>Kilimo AI is a multilingual agricultural assistant that uses Retrieval-Augmented Generation (RAG) and voice interaction to help farmers get crop guidance based on curated farming knowledge.</p>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/Yuriamani/RAG-chatbot.git
cd RAG-chatbot
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Add OpenAI API Key</h3>
Add a .env file and add your openAI/ groq API Key

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python ingest_database.py
python chatbot.py
```
