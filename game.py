from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys
import pdbp
import time
from langchain import hub
import re

# use deepseek model as sidekick
# use gemma model for story generation


example_prompt = """You are a creative and engaging game master chatbot. Based on a player's prompt, generate an immersive scenario and offer a set of logical and exciting options. Keep answers concise, descriptive, and interactive. Follow the tone and format of the examples below.

Example 1:
Q: I want to explore a mysterious forest at night.
A: You step into the dense forest, the moonlight barely breaking through the canopy. Every rustle of leaves sends a chill down your spine. Somewhere in the distance, you hear a low growl.
Options:
1. Draw your sword and move toward the sound.
2. Climb a nearby tree to scout the area.
3. Light a torch and cautiously move forward.
4. Retreat quietly and head back to camp.

Example 2:
Q: I want to sneak into a noble's estate during a banquet.
A: The estate is buzzing with music and laughter. Guards are stationed at the front, but a servant's entrance on the side appears unguarded. You spot a ladder leaning against a balcony above.
Options:
1. Disguise yourself as a servant and walk in.
2. Climb the ladder to access the second floor.
3. Create a distraction in the courtyard.
4. Wait for a guest to arrive and slip in behind them.

Example 3:
Q: I want to challenge the local champion to a duel in the arena.
A: The crowd roars as you enter the sandy arena. Across from you stands the towering champion, wielding a heavy battle axe. The announcer calls for silence. The duel is about to begin.
Options:
1. Charge aggressively and strike first.
2. Dodge and study their fighting style.
3. Attempt to taunt the champion and break their focus.
4. Offer a handshake before the fight begins.

Now answer the user's question below in the same tone and style.

Q: {input}
A:"""


def get_story_prompt():
    base_prompt = hub.pull("hwchase17/react")
    return base_prompt.partial(prefix=example_prompt)


PROMPT = get_story_prompt()

# Initialize the Ollama LLM
DEEPSEEK_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
GEMMA_MODEL = OllamaLLM(model="gemma3:1b")

api_tools = []

def extract_output(input_text: str):
    """ Function to extract output after <think></think> tags from Agent's output response. """
    match = re.search(r"</think>(.*)", input_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_text.strip()

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

    print(f"\nGame Master: {GEMMA_MODEL.invoke(initial_prompt)}")
    # time.sleep(10)
    # print("\nYou can type 'sidekick' to ask for advice or 'exit' to quit the game.")
    memory = ChatMessageHistory(session_id=1)
    agent = create_react_agent(llm=GEMMA_MODEL, tools=api_tools, prompt=PROMPT)
    agent_executor = AgentExecutor(agent=agent, 
                        tools=api_tools, 
                        handle_parsing_errors=True,
                        max_iterations=20, # Increased number of max iterations to avoid time out
                        max_execution_time=120) # Increased the maximum timeout to allow for longer processing) 

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda _: ChatMessageHistory(session_id=1),
        input_messages_key="input",  # conversation_id?
        history_messages_key="chat_history"
    )

    while True:
        user_input = input("\nWhat do you do? ")
        if user_input.lower() == "exit":
            print("\nThanks for playing! Goodbye!")
            break

        if "sidekick" in user_input.lower():
            advice = sidekick_advice(user_input)
            print(f"\nSidekick: {advice}")
            continue

        # response = GEMMA_MODEL.invoke(user_input)  # Use LangChain's conversation chain
        raw_response = agent_with_chat_history.invoke({"input": user_input}, {"configurable": {'session_id': 1}})
        response = extract_output(raw_response.get("output", ""))
        time.sleep(10)
        print(f"\nGame Master: {response}")


# Image description
def describe_image(image_path: str) -> str:
    """Use Qwen-VL to describe an image and return the description."""

    # TODO: implement this function


    
    response = ollama.generate(model="qwen-vl", prompt=f"Describe this image: {image_path}")
    time.sleep(10)
    print(response)


# Sidekick advice
def sidekick_advice(prompt: str) -> str:
    """Ask the sidekick for advice based on the player's prompt."""
    advice = DEEPSEEK_MODEL.invoke(f"Give the player advice about: {prompt}")
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