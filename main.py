import asyncio
import datetime
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from browser_use import Agent, Browser  # бібліотека повинна бути в проєкті

load_dotenv()
logging.basicConfig(level=logging.INFO)
browser = Browser()
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
gemini = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.7
)

LOGIN_PROMPT = HumanMessage(
    content="""
    Open https://www.saucedemo.com

    1. Enter 'standard_user' in the Username field
    2. Enter 'secret_sauce' in the Password field
    3. Click the Login button
    4. Wait for the page to load and check if the URL contains 'inventory.html'
    5. If successful, print 'Login passed'
    """,
    additional_kwargs={"scenario_name": "login_test"}
)

ADD_TO_CART_PROMPT = HumanMessage(
    content="""
    Open https://www.saucedemo.com

    1. Log in using username 'standard_user' and password 'secret_sauce'
    2. Click 'Add to cart' on the first product
    3. Confirm the cart icon now shows '1'
    4. Click the cart icon
    5. Confirm that at least one item is listed in the cart
    6. If everything works, print 'Cart test passed'
    """,
    additional_kwargs={"scenario_name": "add_to_cart_test"}
)

async def run_agent(llm, model_name: str, prompt):
    logging.info(f"Running test: {prompt.additional_kwargs['scenario_name']} with model: {model_name}")
    agent = Agent(
        task=prompt.content,
        llm=llm,
        browser=browser,
        generate_gif=True,
        use_vision=True,
        save_conversation_path=f"e2elogs/{model_name}/{prompt.additional_kwargs['scenario_name']}/{TIMESTAMP}"
    )
    await agent.run()
    input("Press Enter to close the browser...")
    await browser.close()

if __name__ == '__main__':
    # asyncio.run(run_agent(gemini, "gemini", LOGIN_PROMPT))
    asyncio.run(run_agent(gemini, "gemini", ADD_TO_CART_PROMPT))
