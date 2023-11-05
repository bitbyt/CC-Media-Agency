import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

import json
import http.client
import base64
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMMathChain, RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper

import replicate

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Load default config
PRO_GPT_MODEL = os.getenv("PRO_GPT_MODEL")
BASE_GPT_MODEL = os.getenv("BASE_GPT_MODEL")
SUB_GPT_MODEL = os.getenv("SUB_GPT_MODEL")

# Search Function
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": 10
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    print('Searching for... ', query)

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return response.text

# Website Scrape Function
def scrape(url: str):
    print('Scraping website... ', url)

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Parse the data
    data = json.dumps({"url": url})

    # Send the POST request
    browserless_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(browserless_url, headers=headers, data=data)

    if response.status_code == 200:
         # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        print("Scraped content:", text)

        # Content might be really long and hit the token limit, we should summarize the text
        if len(text) > 10000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print("HTTP request failed with status code {response.status_code}")


# Summarise Function
def summary(content):
    # invoke chatGPT
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # Use LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    
    documents = text_splitter.create_documents([content])

    # Reusable prompt for each content on the split chain
    map_prompt = """
    Summarize of the following text for research purpose:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    # Run the summary chain
    output = summary_chain.run(input_documents=documents)

    return output

# def get_image_name():
#     image_count = cl.user_session.get("image_count")
#     if image_count is None:
#         image_count = 0
#     else:
#         image_count += 1

#     cl.user_session.set("image_count", image_count)

#     return f"image-{image_count}"

# Image generator
def generate_image(prompt):
    # Use the 'replicate' library to run an AI model for text-to-image generation
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={
            "prompt": prompt
        }
    )

    if output and len(output) > 0:
        # Get the image URL from the output
        image_url = output[0]
        print(f"Generated image for '{prompt}': {image_url}")

        # Download the image and save it with a filename based on the prompt and current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # shortened_prompt = prompt[:50]

        # Download the image using 'requests' library and save it to a file
        response = requests.get(image_url)
        if response.status_code == 200:
            file = os.path.basename(image_url).replace(".png", "")
            name = f"{file}_{current_time}"

            if not os.path.exists("./image"):
                os.mkdir("./image")
            filename = f"./image/{name}.png"

            with open(filename, "wb") as file:
                file.write(response.content)
            return f"Image saved as '{filename}'"
        else:
            return "The image could not be successfully downloaded and saved."
    else:
        return "The image generation process was unsuccessful."

# Image reviewer
def review_image(image_path, prompt):
    # Use the 'replicate' library to run an AI model for image review
    output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={
            "image": open(image_path, "rb"),
            "prompt": f"Please provide a description of the image and then rate, on a scale of 1 to 10, how closely the image aligns with the provided description. {prompt}?",
        }
    )

    # Concatenate the output into a single string and return it
    result = ""
    for item in output:
        result += item
    return result

# Researcher
def research(query):
    tools = [    
        Tool(
            name = "search",
            func = search,
            description = "Use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),          
        Tool(
            name = "scrape",
            func = scrape,
            description = "Use this to load content from a website url"
        ),   
    ]

    llm = ChatOpenAI(temperature=0, model=BASE_GPT_MODEL)

    system_message = SystemMessage(
        content="""You are a world-class researcher dedicated to factual accuracy and thorough data gathering. You do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the query
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After searching and scraping, you should think "can I increase the research quality by searching and scraping for something new?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered. 
            5/ You should not make things up, you should only write facts & data that you have gathered.
            6/ In the final output, You should include all reference data & links to back up your research."""
    )

    agent_kwargs = {
        "system_message": system_message,
    }

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    results = agent.run(query)
    return results

# Researcher V2
def research_v2(query):
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./chroma"
    )

    search = GoogleSearchAPIWrapper()

    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )

    return results