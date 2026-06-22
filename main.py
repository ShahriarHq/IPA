import asyncio
import configparser
from langchain_openai import ChatOpenAI
from browser_use import Agent, ChatOpenAI

# Load config from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

API_KEY   = config["openrouter"]["api_key"]
MODEL     = config["openrouter"]["google_free_model"]
BASE_URL  = config["openrouter"]["base_url"]
TEMP      = float(config["openrouter"]["temperature"])

async def main():
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=TEMP,
    )
    agent = Agent(
        task=(
            "Go to en.prothomalo.com, "
            "find the sports post, and collect the last 3 hours post updates,"
            "collect those updates and generate a text file.,"
        ),
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())