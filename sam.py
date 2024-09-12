from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import SystemMessage, HumanMessage
from langchain.llms import OpenAI

# Initialize the LLM (replace with your own model, e.g., OpenAI)
llm = OpenAI()

# Function to load previous chat history from a file or DB (for simplicity, using a text file)
def load_previous_summary():
    try:
        with open("chat_summary.txt", "r") as file:
            summary = file.read()
        return summary
    except FileNotFoundError:
        return ""  # If no previous summary exists

# Save updated summary after each conversation
def save_summary(summary):
    with open("chat_summary.txt", "w") as file:
        file.write(summary)

# Load the previous summary into memory
previous_summary = load_previous_summary()

# Initialize memory with the loaded summary (if any)
memory = ConversationSummaryBufferMemory(llm=llm, initial_summary=previous_summary)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt_template=[
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability. Your response is directly rendered in markdown. Response should be only in markdown. Every time wish the user in a unique way."),
    ]
)

async def generate_response(text):
    # Append the new human message to the conversation
    human_message = HumanMessage(content=text)

    # Get the response from the conversation chain
    response = await conversation.acall(inputs={"human_input": human_message.content})
    
    # Save the updated summary
    save_summary(memory.load_memory_variables()["summary"])
    
    # Return the latest response
    return response

# Example usage:
# result = await generate_response("What is the weather today?")
