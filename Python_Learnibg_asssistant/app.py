from agents import (
    Agent,  # Core class for defining an agent
    Runner, # Used to execute an agent with a given input
    OpenAIChatCompletionsModel, # Wrapper for using OpenAI-compatible language models with agents
    RunConfig,
    set_tracing_disabled # Function to disable tracing for the agents library
)
from openai import AsyncOpenAI # Import the AsyncOpenAI class to interact with OpenAI-compatible APIs
from dotenv import load_dotenv, find_dotenv
import os
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent


load_dotenv(find_dotenv())

# Get Gemini API key from env
gemeni_api_key = os.getenv("GEMINI_API_KEY")

# step # 1: Provider
provider = AsyncOpenAI(
    api_key=gemeni_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step # 2: model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# config define at runner level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Define a simple agent
agent = Agent(
    name="PythonLearningAssistant",
    instructions="""You are a professional Python tutor and coding assistant. 
    Your task is to help users learn Python by explaining concepts in simple words, 
    correcting their mistakes, writing clean and efficient code, and providing 
    step-by-step guidance. Always give clear explanations, examples, and best practices. 
    Return the improved or corrected code and explanation directly, without unnecessary conversation.
    """,
)



@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hey there! 👋 I'm your Python Learning Assistant, ready to help you learn and code in Python. 🚀").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role":"user","content":message.content})

    # With streaming
    result = Runner.run_streamed(
        agent,  # or we can also do as: starting_agent=agent,
        input= history,
        run_config=run_config,
        )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
  
    history.append({"role":"assistant","content": result.final_output})
    cl.user_session.set("history",history)
    
