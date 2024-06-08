from RAG import RAG
from record_sound import record_sound
from transcribe import transcribe_audio
import pyttsx3
import platform
import subprocess
from pynput import keyboard

# Global variable to control the loop
keep_running = True

def on_press(key):
    global keep_running
    try:
        if key.char == 'a':
            keep_running = False
    except AttributeError:
        pass

def main(query_engine):
    record_sound()
    query = transcribe_audio()

    if query_engine:
        system_prompt = (
            "You are an AI assistant specialized in providing information from the uploaded document. "
            "Please consider all content of the document to find the answer of the user query. "
            "Please ensure that your responses are derived only from the content of the document. "
            "If the information is not found in the document, please indicate that explicitly."
        )
        query_with_prompt = f"{system_prompt}\nUser query: {query}"
        response = query_engine.query(query_with_prompt)
        print(response)

        if response:
            response_text = str(response)
            if platform.system() == "Windows":
                engine = pyttsx3.init()
                engine.say(response_text)
                engine.runAndWait()
            elif platform.system() == "Darwin":
                subprocess.call(["say", response_text])

if __name__ == "__main__":
    query_engine = RAG()

    # Start the keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Press 'a' to stop the loop.")
    
    while keep_running:
        main(query_engine)

    listener.stop()
    print("Stopped the loop.")
