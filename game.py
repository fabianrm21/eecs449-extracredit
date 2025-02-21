import ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=deepseek_model, memory=memory)

# main game
def game_loop():
    print("Welcome to the adventure!")
    while True:
        user_input = input("What do you do? ")
        response = ollama.generate(model="deepseek", prompt=user_input)
        print(response)
        if user_input.lower() == "exit":
            break

# image description
response = ollama.generate(model="qwen-vl", prompt="Describe this image: [image_path]")
print(response)

# sidekick advice
def sidekick_advice():
    advice = ollama.generate(model="deepseek", prompt="Give the player advice.")
    print(f"Sidekick: {advice}")

if __name__ == "__main__":
    game_loop()