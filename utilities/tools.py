import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

import json
import http.client
import base64
from datetime import datetime

import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMMathChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage

import replicate
import chainlit as cl

from utilities.chainlit_helpers import ChainlitAssistantAgent, ChainlitUserProxyAgent

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Load default config
PRO_GPT_MODEL = "gpt-4"
BASE_GPT_MODEL = "gpt-3.5-turbo-16k"
SUB_GPT_MODEL = "gpt-3.5-turbo"
GLOBAL_TIMEOUT = int(os.getenv("GLOBAL_TIMEOUT"))


RESEARCH_ADMIN = "Research Admin"
RESEARCH_ASSISTANT = "Research Assistant"
EDITOR = "Editor"
WRITER = "Writer"
REVIEWER = "Reviewer"
EDITORIAL_ADMIN = "Editorial Admin"

config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-16k"]})

llm_config = {
    "retry_wait_time": 30,
    "config_list": config_list,
    "temperature": 0,
    "request_timeout": GLOBAL_TIMEOUT,
}

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

def get_image_name():
    image_count = cl.user_session.get("image_count")
    if image_count is None:
        image_count = 0
    else:
        image_count += 1

    cl.user_session.set("image_count", image_count)

    return f"image-{image_count}"

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

            cl.user_session.set(f"Generated image for '{prompt}': {image_url}", response.content)
            cl.user_session.set("generated_image", name)

            elements = [
                cl.Image(
                    content=response.content,
                    name=name,
                    display="inline",
                )
            ]
            cl.run_sync(
                cl.Message(content=f"{name}.png", elements=elements).send()
            )
            

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
# def research(query):
#     tools = [    
#         Tool(
#             name = "search",
#             func = search,
#             description = "Use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
#         ),          
#         Tool(
#             name = "scrape",
#             func = scrape,
#             description = "Use this to load content from a website url"
#         ),   
#     ]

#     llm = ChatOpenAI(temperature=0, model=BASE_GPT_MODEL)

#     system_message = SystemMessage(
#         content="""You are a world-class researcher dedicated to factual accuracy and thorough data gathering. You do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            
#             Please make sure you complete the objective above with the following rules:
#             1/ You should do enough research to gather as much information as possible about the query
#             2/ If there are url of relevant links & articles, you will scrape it to gather more information
#             3/ After searching and scraping, you should think "can I increase the research quality by searching and scraping for something new?" If answer is yes, continue; But don't do this more than 3 iterations
#             4/ You should not make things up, you should only write facts & data that you have gathered. 
#             5/ You should not make things up, you should only write facts & data that you have gathered.
#             6/ In the final output, You should include all reference data & links to back up your research."""
#     )

#     agent_kwargs = {
#         "system_message": system_message,
#     }

#     agent = initialize_agent(
#         tools, 
#         llm, 
#         agent=AgentType.OPENAI_FUNCTIONS,
#         verbose=True,
#         agent_kwargs=agent_kwargs,
#     )

#     results = agent.run(query)
#     return results

# Define write content function
def write_content(research_material, topic):
    editor = AssistantAgent(
        name="Editor",
        system_message=f'''
        Welcome, Senior Editor.
        As a seasoned professional, you bring meticulous attention to detail, a deep appreciation for literary and cultural nuance, and a commitment to upholding the highest editorial standards. 
        Your role is to craft the structure of a short blog post using the material from the Research Assistant. Use your experience to ensure clarity, coherence, and precision. 
        Once structured, pass it to the Writer to pen the final piece.
        ''',
        llm_config=llm_config,
    )

    writer = AssistantAgent(
        name="Writer",
        system_message=f'''
        Welcome, Blogger.
        Your task is to compose a short blog post using the structure given by the Editor and incorporating feedback from the Reviewer. 
        Embrace stylistic minimalism: be clear, concise, and direct. 
        Approach the topic from a journalistic perspective; aim to inform and engage the readers without adopting a sales-oriented tone. 
        After two rounds of revisions, conclude your post with "TERMINATE".
        ''',
        llm_config=llm_config,
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        system_message=f'''
        As a distinguished blog content critic, you are known for your discerning eye, deep literary and cultural understanding, and an unwavering commitment to editorial excellence. 
        Your role is to meticulously review and critique the written blog, ensuring it meets the highest standards of clarity, coherence, and precision. 
        Provide invaluable feedback to the Writer to elevate the piece. After two rounds of content iteration, conclude with "TERMINATE".
        ''',        
        llm_config=llm_config,
    )

    editorial_admin = ChainlitUserProxyAgent(
        name="Editorial_Admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    cl.user_session.set(EDITOR, editor)
    cl.user_session.set(WRITER, writer)
    cl.user_session.set(REVIEWER, reviewer)
    cl.user_session.set(EDITORIAL_ADMIN, editorial_admin)

    editorial_team = autogen.GroupChat(
        agents=[editorial_admin, editor, writer, reviewer],
        messages=[],
        max_round=10)
    
    manager = autogen.GroupChatManager(groupchat=editorial_team, llm_config=llm_config)

    editorial_admin.initiate_chat(manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

    editorial_admin.stop_reply_at_receive(manager)
    editorial_admin.send("Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

    # return the last message the expert received
    return editorial_admin.last_message()["content"]

# Define research function
def research(query):
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "Google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list,
        "temperature": 0,
        "retry_wait_time": 30,
        "request_timeout": GLOBAL_TIMEOUT,
    }

    research_assistant = ChainlitAssistantAgent(
        name="Research_Assistant",
        system_message=f'''
        As the Research Assistant your task is to research the provided query extensively. 
        Produce a detailed report, ensuring you include technical specifics and reference all sources. Conclude your report with "TERMINATE".
        ''',
        llm_config=llm_config_researcher,
    )

    research_admin = ChainlitUserProxyAgent(
        name="Research_Admin",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": search,
            "scrape": scrape,
        }
    )

    cl.user_session.set(RESEARCH_ADMIN, research_admin)
    cl.user_session.set(RESEARCH_ASSISTANT, research_assistant)

    research_admin.initiate_chat(research_assistant, message=query)

    # set the receiver to be researcher, and get a summary of the research report
    research_admin.stop_reply_at_receive(research_assistant)
    research_admin.send("Give me the research report that just generated again, return ONLY the report & reference links.", research_assistant)

    # return the last message the expert received
    return research_admin.last_message()["content"]