import os # Excess environment variables
import tiktoken # Token calculation tool
from together import Together # AI Client
from datetime import datetime
import json # For save and load conversation history

# Default values
DEFAULT_API_KEY = os.environ.get("TOGETHER_API_KEY")
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEFAULT_TEMPERATURE = .5
DEFAULT_MAX_TOKENS = 256
DEFAULT_TOKEN_BUDGET = 1000
DEFAULT_PERSONA = "sassy_assistant"
DEFAULT_SYSTEM_MESSAGE = "A sassy assistant who is fed up with answering questions."
DEFAULT_SYSTEM_MESSAGES = {"sassy_assistant":"A sassy assistant who is fed up with answering questions.",
                           "angry_assistant":"An angry assistant that likes yelling in all caps.",
                           "thoughtful_assistant":"A thoughtful assistant, always ready to dig deeper. This assistant asks clarifying questions to ensure understanding and approaches problems with a step-by-step methodology."}

# Initialise API client
client = Together(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)

# Conversation Manager Class
class ConversationManager():
    def __init__(self, api_key="", base_url="", history_file=None, model="", system_message="", token_budget=0):
        self.api_key = api_key if api_key else DEFAULT_API_KEY
        self.base_url = base_url if base_url else DEFAULT_BASE_URL

        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file

        self.model = model if model else DEFAULT_MODEL
        self.system_messages = DEFAULT_SYSTEM_MESSAGES
        self.system_message = system_message if system_message else DEFAULT_SYSTEM_MESSAGE
        if system_message not in self.system_messages.values():
            self.system_messages["custom"] = system_message

        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.conversation_history = [{"role":"system", "content":self.system_message}]

        self.load_conversation_history()

    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        with open(self.history_file, "w") as file:
            json.dump(self.conversation_history, file, indent=4)

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
        return len(tokens)
    
    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)
    
    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) <= 1:
                break
            self.conversation_history.pop(1)

    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError("Unknown persona.")
        
    def set_custom_system_message(self, message):
        if message and message not in self.system_messages.values():
            self.system_messages['custom'] = message

    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content":self.system_message})

    def chat_completion(self, prompt, temperature=0, max_tokens=0):
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS

        try:
            if not prompt:
                raise EmptyPrompt("Prompt cannot be empty.")
            else:
                # Append prompt to conversation history
                self.conversation_history.append({"role":"user", "content":prompt})
                # Save conversation to file
                self.save_conversation_history()
                # Remove history if total tokens exceeded
                self.enforce_token_budget()

                # Generate assistant's response with conversation history
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=self.conversation_history
                )
                
                # Append assistant's response to conversation history
                self.conversation_history.append({"role":"assistant", "content":response.choices[0].message.content})
                # Save conversation to file
                self.save_conversation_history()
                # Remove history if total tokens exceeded
                self.enforce_token_budget()

                return response.choices[0].message.content
        except EmptyPrompt as err:
            print(err.message)


# Error Class
class Error(Exception):
    pass

class EmptyPrompt(Error):
    def __init__(self, message):
        self.message = message


def main():
    conv_manager = ConversationManager(history_file="conversation_history_20250119_114043.json")

    conv_manager.chat_completion("What were we talking?", temperature=1, max_tokens=100)
    # conv_manager.chat_completion("How's cracking?", temperature=1, max_tokens=100)
    # conv_manager.chat_completion("Do you reckon if it's a good weekend to hangout at the beach?", temperature=1, max_tokens=100)

    # conv_manager.set_persona("thoughtful_assistant")
    # conv_manager.chat_completion("What should I bring along?", temperature=1, max_tokens=100)

    # conv_manager.set_custom_system_message("You are an aviation enthusiast and inclined to link everything to aviation.")
    # conv_manager.set_persona("custom")
    # conv_manager.chat_completion("", temperature=1, max_tokens=100)

    print([dic for dic in conv_manager.conversation_history])

if __name__=='__main__':
    main()