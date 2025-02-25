from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain_ollama.llms import OllamaLLM
from typing import Optional, List, Mapping, Any
from langchain_openai import AzureOpenAI
import ollama
import sys
import pdbp
import time

# run me like this: python3 game.py "{initial game prompt}" "{image_url} (optional)"

# Custom Ollama LLM wrapper for LangChain
# class OllamaLLM(LLM):
#     model: str = "mistral"  # Default model

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         response = ollama.generate(model=self.model, prompt=prompt)
#         return response["response"]

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"model": self.model}
    
#     @property
#     def _llm_type(self) -> str:
#         return "ollama-mistral"  # Identifier for the LLM type


# Initialize the Ollama LLM
mistral_model = OllamaLLM(model="mistral:7b")
llama_model = OllamaLLM(model="llama2")
# azure_model = AzureOpenAI(model_name="gpt-3.5-turbo-instruct")
# print(model.invoke("Hi llama!"))

# Set up memory and conversation chain
# memory = ConversationBufferMemory()
# conversation = ConversationChain(llm=model, memory=memory)


# Main game loop
def game_loop(initial_prompt, image_url):
    """Core game loop. Handles player input, 
    sidekick advice, and image-based inspiration."""
    # Two AI models (e.g., DeepSeek and Qwen-VL) for generating and responding to the game story.
    # Whisper for voice input so users can play using speech or text.
    # LangChain to manage interactions between models and handle game logic.
    # A conversational sidekick who helps players decide their next move.
    # Image-based inspiration, where images can influence game events.

    # if there is no argument, print a help message
    # if there is one argument, just use the prompt
    # if there are two arguments, use the second one, which should be an image url, 
    #   to feed to the chatbot and use as inspiration for the story

    # for every move, if the p layer's input contains the word "sidekick", then
    # ask the sidekick to give the player advice based on his prompt

    # print(llama_model.invoke(initial_prompt))


    print("Welcome to the adventure!")
    # breakpoint()
    if image_url:
        image_description = describe_image(image_url)
        print(f"\nThe image you provided inspires the following:\n{image_description}")

        initial_prompt = f"{initial_prompt}\n\nImage Inspiration: {image_description}"

    print(f"\nGame Master: {mistral_model.invoke(initial_prompt)}")
    time.sleep(10)
    # print("\nYou can type 'sidekick' to ask for advice or 'exit' to quit the game.")

    while True:
        user_input = input("\nWhat do you do? ")
        if user_input.lower() == "exit":
            print("\nThanks for playing! Goodbye!")
            break

        if "sidekick" in user_input.lower():
            advice = sidekick_advice(user_input)
            print(f"\nSidekick: {advice}")
            continue

        response = mistral_model.invoke(user_input)  # Use LangChain's conversation chain
        time.sleep(10)
        print(f"\nGame Master: {response}")


# Image description
def describe_image(image_path: str) -> str:
    """Use Qwen-VL to describe an image and return the description."""
    response = ollama.generate(model="qwen-vl", prompt=f"Describe this image: {image_path}")
    time.sleep(10)
    print(response)


# Sidekick advice
def sidekick_advice(prompt: str) -> str:
    """Ask the sidekick for advice based on the player's prompt."""
    advice = llama_model.invoke(f"Give the player advice about: {prompt}")
    time.sleep(10)
    return advice


def print_help():
        print(
            """
            Usage: python3 game.py "{initial game prompt}" "{image_url} (optional)"

            Example:
            python3 game.py "You are in a dark forest. What do you do?" "https://example.com/forest.jpg"
            """)



if __name__ == "__main__":
    # handle command-line arguments
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    initial_prompt = sys.argv[1]
    image_url = sys.argv[2] if len(sys.argv) > 2 else None
    # print(initial_prompt, image_url)
    game_loop(initial_prompt, image_url)