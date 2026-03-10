from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# voice chat req...
import os
import gradio as gr
from gtts import gTTS
from groq import Groq
from tempfile import NamedTemporaryFile

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
# CHROMA_PATH = r"chroma_db"
CHROMA_PATH = r"huggingface_chroma_db"

# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# initiate the model
# llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# Initialize standard Groq client for Whisper
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- VOICE FUNCTIONS ---
def transcribe_audio(audio_file_path):
    # Open the audio file
    with open(audio_file_path, "rb") as file:
        # Use Groq's whisper-large-v3 model
        transcription = client.audio.transcriptions.create(
            file=(audio_file_path, file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    return transcription

def speech_to_text(audio_path):
    transcription = transcribe_audio(audio_path) # ["text"]
    return transcription

def text_to_speech(text):
    tts = gTTS(text)
    output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(output_audio.name)
    return output_audio.name

# call this function for every message added to the chatbot
def stream_response(message, audio_input, history):
    if history is None:
        history = []
    #print(f"Input: {message}. History: {history}\n")
    # Determine the actual query (Voice or Text)
    if message and message.strip() != "":
        query = message
        audio_input = None # Ignore the ghost audio
    elif audio_input:
        query = speech_to_text(audio_input)
        # Check for Whisper hallucinations (static/silence)
        # If Whisper hears static, it often returns these specific phrases
        hallucinations = ["Вот", "аркава", "Thank you", "字幕", "Screencast"]
        if any(h in query for h in hallucinations) or len(query.strip()) < 2:
             yield history, None, "", None
             return
    else:
        # yield history + [("System", "Please type something or record audio.")], None, ""
        return
    # query = message
    # if audio_input:
    #     query = speech_to_text(audio_input)
    
    # if not query:
    #     yield "Please provide text or audio input.", None
    #     return

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(query)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if query is not None:

        partial_message = ""

        history.append({"role": "user", "content": query})
        
        # Add a placeholder for the Assistant's message that we will update
        history.append({"role": "assistant", "content": ""})

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {query}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            history[-1]["content"] = partial_message
            # new_history = history + [(query, partial_message)]
            # Yielding 4 values: [history, audio, txt_input, audio_input]
            yield history, None, "", None
        # Generate Audio once text is finished
        try:
            audio_path = text_to_speech(partial_message)
            # FINAL YIELD: Send the audio and CLEAR the audio_input component (None)
            yield history, audio_path, "", None
        except Exception as e:
            print(f"TTS Error: {e}")
            yield history, None, "", None

# initiate the Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# 🌾 Kilimo AI – Multilingual Voice Assistant for Farmers")
    
    chatbot = gr.Chatbot(label="Conversation History", height=400)
    
    with gr.Row():
        # Textbox with a high scale to take up most of the row
        txt_input = gr.Textbox(
            show_label=False, 
            placeholder="Ask about your crops...", 
            scale=7
        )
        # Small send button with an icon
        submit_btn = gr.Button("➤", variant="primary", scale=1, min_width=50)

    with gr.Row():
        # Audio input stays below or alongside
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
        clear_btn = gr.ClearButton([txt_input, audio_input, chatbot], value="🗑️ Clear Chat")

    audio_output = gr.Audio(interactive=False, visible=True, autoplay=True)

    # --- BUTTON TRIGGERS ---
    # We use the same function as before, but triggered by the icon button
    output_list = [chatbot, audio_output, txt_input, audio_input]
    input_list = [txt_input, audio_input, chatbot]

    submit_btn.click(
        fn=stream_response, 
        inputs=input_list, 
        outputs=output_list
    )
    
    # Still allow "Enter" key to send
    txt_input.submit(
        fn=stream_response, 
        inputs=input_list, 
        outputs=output_list
    )

demo.launch(theme=gr.themes.Glass())

# with gr.Blocks() as demo:
#     gr.Markdown("# 🌾 Kilimo Voice RAG Assistant")
    
#     with gr.Row():
#         with gr.Column(scale=4):
#             # Standard ChatInterface
#             chat = gr.ChatInterface(
#                 fn=stream_response,
#                 additional_inputs=[
#                     gr.Audio(type="filepath", label="Or Speak your question")
#                 ],
#                 # We define the outputs to handle both the text stream and final audio
#                 additional_outputs=[
#                     gr.Audio(label="AI Voice Response", autoplay=True)
#                 ]
#             )

# demo.launch(share=True)

# chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
#     container=False,
#     autoscroll=True,
#     scale=7),
#     additional_inputs=[
#         gr.Audio(type="filepath", label="Speak")
#     ],
#     additional_outputs=[gr.Textbox(label="Response Text"),
#         gr.Audio(label="Response Audio")],
# save_history=True)

# # launch the Gradio app
# chatbot.launch()