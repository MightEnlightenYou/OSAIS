from tkinter import Tk, Frame, Scrollbar, Text, Button, END, Label, Entry, filedialog
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LLMModel:
    def __init__(self, model_path, n_gpu_layers, n_batch, n_ctx, max_tokens, f16_kv, verbose, temperature, top_k, top_p, repetition_penalty, length_penalty, threads, max_length):
        self.model_path = model_path
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_batch = int(n_batch)
        self.n_ctx = int(n_ctx)
        self.max_tokens = int(max_tokens)
        self.f16_kv = bool(f16_kv)
        self.verbose = bool(verbose)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.length_penalty = float(length_penalty)
        self.threads = int(threads)
        self.max_length = int(max_length)

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ctx=self.n_ctx,
            max_tokens=self.max_tokens,
            f16_kv=self.f16_kv,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=self.verbose,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            threads=self.threads,
            max_length=self.max_length,
        )

    def generate_response(self, prompt):
        return self.llm(prompt)

# Create a Tkinter window
window = Tk()
window.title("LLM Chat Application")
window.geometry("800x400")

# Create a Frame for the chat box
chat_frame = Frame(window, bd=2, relief="groove")
chat_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Create a Text widget for displaying the chat messages
chat_box = Text(chat_frame, bd=0, bg="white", height="8", width="50", font="Arial")
chat_box.config(state="disabled")
chat_box.pack(side="left", fill="both", expand=True)

# Create a Scrollbar for the chat box
scrollbar = Scrollbar(chat_frame, command=chat_box.yview)
scrollbar.pack(side="right", fill="y")
chat_box["yscrollcommand"] = scrollbar.set

# Create a Frame for the user input and send button
input_frame = Frame(window, bd=2, relief="groove")
input_frame.pack(side="bottom", fill="x", padx=10, pady=10)

# Create an Entry widget for user input
user_input = Text(input_frame, bd=0, bg="white", width="29", height="5", font="Arial")
user_input.pack(side="left", fill="both", expand=True)

# Create a Send button
send_button = Button(input_frame, text="Send")
send_button.pack(side="right", padx=5)

# Create a Frame for the sidebar
sidebar_frame = Frame(window, bd=2, relief="groove")
sidebar_frame.pack(side="right", fill="y", padx=10, pady=10)

# Create labels and entries for LLMModel parameters in the sidebar
param_labels = [
    "Model Path",
    "n_gpu_layers",
    "n_batch",
    "n_ctx",
    "max_tokens",
    "f16_kv",
    "verbose",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "length_penalty",
    "threads",
    "max_length",
]
param_entries = []

def browse_model_path():
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.GGUF")])
    model_path_entry.delete(0, END)
    model_path_entry.insert(END, model_path)

for label in param_labels:
    param_label = Label(sidebar_frame, text=label + ":")
    param_label.pack(anchor="w")

    param_entry = Entry(sidebar_frame, width=30)
    param_entry.pack(anchor="w")

    param_entries.append(param_entry)

# Set default values for each parameter
default_values = {
    "Model Path": "",
    "n_gpu_layers": "35",
    "n_batch": "512",
    "n_ctx": "4096",
    "max_tokens": "1000",
    "f16_kv": "True",
    "verbose": "True",
    "temperature": "0.8",
    "top_k": "0",
    "top_p": "1.0",
    "repetition_penalty": "1.0",
    "length_penalty": "1.0",
    "threads": "4",
    "max_length": "200",
}

for i, label in enumerate(param_labels):
    param_entries[i].insert(END, default_values[label])

# Create a Label and Entry for the model path in the sidebar
model_path_label = Label(sidebar_frame, text="Model Path:")
model_path_label.pack(anchor="w")

model_path_entry = Entry(sidebar_frame, width=30)
model_path_entry.pack(side="left", anchor="w")

browse_button = Button(sidebar_frame, text="Browse", command=browse_model_path)
browse_button.pack(side="left", padx=5)

# Function to handle sending user input and generating LLM response
def send_message(event):
    user_message = user_input.get("1.0", END).strip()
    user_input.delete("1.0", END)
    chat_box.config(state="normal")
    chat_box.insert(END, "User: " + user_message + "\n")
    chat_box.config(state="disabled")

    # Retrieve LLM settings from entry fields
    llm_settings = {}
    for i, label in enumerate(param_labels[1:]):
        llm_settings[label] = param_entries[i+1].get()

    # Add n_gpu_layers to llm_settings
    llm_settings["n_gpu_layers"] = param_entries[1].get()

    # Instantiate the LLMModel with the provided settings and model path
    llm_model = LLMModel(
        model_path=model_path_entry.get(),
        **llm_settings
    )

    combined_prompt = user_message
    response = llm_model.generate_response(combined_prompt)

    chat_box.config(state="normal")
    chat_box.insert(END, "LLM: " + response)
    chat_box.config(state="disabled")
    chat_box.see(END)

# Bind the Enter key to the send_message function
window.bind("<Return>", send_message)

# Run the Tkinter event loop
window.mainloop()