import tiktoken
from together import Together
import os

# Default values
DEFAULT_API_KEY = os.environ.get("TOGETHER_API_KEY")
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEFAULT_TEMPERATURE = .5
DEFAULT_MAX_TOKENS = 256
DEFAULT_TOKEN_BUDGET = 1000
DEFAULT_SYSTEM_MESSAGES = {"sassy_assistant":"A sassy assistant who is fed up with answering questions.",
                           "angry_assistant":"An angry assistant that likes yelling in all caps.",
                           "thoughtful_assistant":"A thoughtful assistant, always ready to dig deeper. This assistant asks clarifying questions to ensure understanding and approaches problems with a step-by-step methodology."}

# Initialise API client
client = Together(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)

# Conversation Manager Class
class ConversationManager():
    def __init__(self, api_key="", base_url="", model="", system_message="", token_budget=0):
        self.api_key = api_key if api_key else DEFAULT_API_KEY
        self.base_url = base_url if base_url else DEFAULT_BASE_URL
        self.model = model if model else DEFAULT_MODEL

        self.system_messages = DEFAULT_SYSTEM_MESSAGES
        if system_message:
            self.system_messages["custom"] = system_message

        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.history = [{"role":"system", "content":self.system_message}]

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
        return len(tokens)
    
    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.history)
    
    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.history) <= 1:
                break
            self.history.pop(1)    

    def chat_completion(self, prompt, temperature=0, max_tokens=0):
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS

        # Append prompt to conversation history
        self.history.append({"role":"user", "content":prompt})
        # Remove history if total tokens exceeded
        self.enforce_token_budget()

        # Generate assistant's response with conversation history
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=self.history
        )
        
        # Append assistant's response to conversation history
        self.history.append({"role":"assistant", "content":response.choices[0].message.content})
        # Remove history if total tokens exceeded
        self.enforce_token_budget()

        return response.choices[0].message.content
    

def main():
    conv_manager = ConversationManager()
    conv_manager.chat_completion("How's cracking?", temperature=1, max_tokens=10)
    conv_manager.chat_completion("Do you reckon if it's a good weekend to hangout at the beach?", temperature=1, max_tokens=10)

    print(conv_manager.history)

if __name__=='__main__':
    main()