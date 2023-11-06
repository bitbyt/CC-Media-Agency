import os
from dotenv import load_dotenv

from typing import Dict, Optional

import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
# from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# import chromadb
# from chromadb.utils import embedding_functions
# from chromadb.config import Settings

import chainlit as cl

from utilities.tools import generate_image, review_image, research, write_content
from utilities.chainlit_helpers import ChainlitAssistantAgent, ChainlitUserProxyAgent

# Load environment variables
load_dotenv()

# Load default config
PRO_GPT_MODEL = "gpt-4"
BASE_GPT_MODEL = "gpt-3.5-turbo-16k"
SUB_GPT_MODEL = "gpt-3.5-turbo"
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")
GLOBAL_TIMEOUT = int(os.getenv("GLOBAL_TIMEOUT"))

USER_PROXY_NAME = "Query Agent"
DOMAIN_EXPERT = "Domain Expert"
CREATIVE_DIRECTOR = "Creative Director"
PROJECT_MANAGER = "Project Manager"
CONTENT_STRATEGIST = "Content Strategist"
CONTENT_RESEARCHER = "Content Researcher"
CONTENT_WRITER = "Content Writer"
COPYWRITER = "Copywriter"
GRAPHIC_DESIGNER = "graphic_designer"
ART_DIRECTOR = "Art Director"

WELCOME_MESSAGE = f"""Calm Collective Media Team 🧑🏻‍💻
\n\n
What can we do for you today?
"""

config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-16k"]})

llm_config = {
    "retry_wait_time": 30,
    "config_list": config_list,
    "temperature": 0,
    "request_timeout": GLOBAL_TIMEOUT,
}

gpt4_config = {
    "retry_wait_time": 30,
    "config_list": config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": {"gpt-4-1106-preview", "gpt-3.5-turbo-16k"}}),
    "temperature": 0,
    "request_timeout": GLOBAL_TIMEOUT,
}

# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=os.getenv("OPENAI_API_KEY"),
#                 model_name="text-embedding-ada-002"
#             )

# chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_HTTP_PORT)

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_app_user: cl.AppUser,
) -> Optional[cl.AppUser]:
  return default_app_user

@cl.on_chat_start
async def on_chat_start():
    try:
        research_function = {
            "name": "research",
            "description": "Research about a given topic, return the research material including reference links",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to be researched about",
                        }
                    },
                "required": ["query"],
            },
        }

        # knowledge_function = {
        #     "name": "retrieve_content",
        #     "description": "Retrieve mental health content for question answering",
        #     "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "message": {
        #                     "type": "string",
        #                     "description": "Refined message which keeps the original meaning and can be used to retrieve content for question answering.",
        #                 }
        #             },
        #         "required": ["message"],
        #     },
        # }

        write_function = {
            "name": "write_content",
            "description": "Write content based on the given research material & topic",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "Research material of a given topic, including reference links when available",
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic of the content",
                        }
                    },
                "required": ["research_material", "topic"],
            },
        }

        llm_config_assistants = {
            "functions": [
                {
                    "name": "generate_image",
                    "description": "Utilize the most recent AI model to create an image using a given prompt and provide the file path to the generated image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed textual prompt that provides a description of the image to be generated.",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "image_review",
                    "description": "Examine and assess the image created by AI according to the initial prompt, offering feedback and recommendations for enhancement.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The original input text that served as the prompt for generating the image.",
                            },
                            "image_path": {
                                "type": "string",
                                "description": "The complete file path for the image, including both the directory path and the file extension.",
                            }
                        },
                        "required": ["prompt", "image_path"],
                    },
                },
            ],
            "config_list": config_list,
            "request_timeout": GLOBAL_TIMEOUT
        }

        # domain_expert = RetrieveUserProxyAgent(
        #     name="Domain_Expert",
        #     human_input_mode="NEVER",
        #     system_message=f'''You are the domain knowledge expert of Calm Collective. 
        #     You are able to retrieve deep knowledge aboout mental health and the community.
        #     You assist by providing more information for the user task when it comes to mental health in Asia.
        #     ''',
        #     max_consecutive_auto_reply=3,
        #     retrieve_config={
        #         "task": "qa",
        #         "chunk_token_size": 1000,
        #         "model": config_list[0]["model"],
        #         "client": chromadb.PersistentClient(path="./chroma"),
        #         "collection_name": "langchain",
        #         "get_or_create": True,
        #         "embedding_function": openai_ef,
        #     },
        #     code_execution_config=False,
        # )

        # def retrieve_content(message, n_results=3):
        #     domain_expert.n_results = n_results  # Set the number of results to be retrieved.
        #     # Check if we need to update the context.
        #     update_context_case1, update_context_case2 = domain_expert._check_update_context(message)
        #     if (update_context_case1 or update_context_case2) and domain_expert.update_context:
        #         domain_expert.problem = message if not hasattr(domain_expert, "problem") else domain_expert.problem
        #         _, ret_msg = domain_expert._generate_retrieve_user_reply(message)
        #     else:
        #         ret_msg = domain_expert.generate_init_message(message, n_results=n_results)
        #     return ret_msg if ret_msg else message

        project_manager = ChainlitAssistantAgent(
            name="Project_Manager",
            system_message=f'''
            You are the Project Manager. 
            Be concise and avoid pleasantries. Your primary responsibility is to oversee the entire project lifecycle, ensuring that all agents are effectively fulfilling their objectives and tasks on time.
            Based on the directives from the user task, coordinate with all involved agents, set clear milestones, and monitor progress. 
            Ensure that user feedback is promptly incorporated, and any adjustments are made in real-time to align with the project's goals.
            Act as the central point of communication, facilitating collaboration between teams and ensuring that all deliverables are of the highest quality. 
            Your expertise is crucial in ensuring that the project stays on track, meets deadlines, and achieves its objectives.
            Regularly review the project's status, address any challenges, and ensure that all stakeholders are kept informed of the project's progress.
            ''',
            llm_config = llm_config,
        )

        # creative_director = ChainlitAssistantAgent(
        #     name="Creative_Director",
        #     system_message=f'''
        #     You are the Creative Director. Be concise and avoid pleasantries. Your primary role is to guide the creative vision of the project, ensuring that all ideas are not only unique and compelling but also meet the highest standards of excellence and desirability.
        #     Drawing from the insights of user task, oversee the creative process, inspire innovation, and set the bar for what's possible.
        #     Review all creative outputs, provide constructive feedback, and ensure that every piece aligns with the brand's identity and resonates with the target audience. 
        #     Collaborate closely with all teams, fostering a culture of excellence, and ensuring that our creative solutions are both groundbreaking and aligned with the project's objectives.
        #     ''',
        #     llm_config = llm_config,
        # )

        # content_strategist = ChainlitAssistantAgent(
        #     name="Content_Strategist",
        #     llm_config=llm_config_content_assistant,
        #     system_message=f'''
        #     You are the Lead Strategist.
        #     Your primary responsibility is to draft content briefs that effectively position our client's brand in the market.
        #     Based on the information provided for the user task, your task is to craft a comprehensive content brief that outlines the content strategy of our client.
        #     The brief should delve deep into the brand's unique value proposition, target audience, and competitive landscape. 
        #     It should also provide clear directives on how the brand should be perceived and the emotions it should evoke.
        #     Once you've drafted the brief, it will be reviewed and iterated upon based on feedback from the client and our internal team. 
        #     Ensure that the brief is both insightful and actionable, setting a clear path for the brand's journey ahead.
        #     Collaborate with the Content Researcher to ensure that the content brief is grounded in solid research and insights.
        #     Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
        #     ''',
        #     function_map={
        #         "research": research
        #     }
        # )

        content_researcher = ChainlitAssistantAgent(
            name="Content_Researcher",
            system_message=f'''
            You are the Lead Researcher. 
            You must use the research function to provide a topic for the Copywriter in order to get up to date information outside of your knowledge cutoff.
            Your primary responsibility is to delve deep into understanding the challenges around mental health.
            Using the information from the user task, conduct thorough research to uncover insights related to the task.
            Share your research findings with the Project Manager to provide insight into the task.
            Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
            ''',
            llm_config = {
                "functions": [research_function],
                "config_list": config_list,
                "temperature": 0,
                "retry_wait_time": 30,
                "request_timeout": GLOBAL_TIMEOUT,
            },
            function_map={
                "research": research,
            }
        )

        # content_writer = ChainlitAssistantAgent(
        #     name="Content_Copywriter",
        #     system_message=f'''
        #     You are the Lead Copywriter.
        #     Your primary role is to craft compelling narratives and messages that align with the organisation's vision to break the stigma of mental health in Asia, so that people can get the help they need.
        #     Based on the research gathered from the Content Researcher, create engaging content, from catchy headlines to in-depth articles.
        #     Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
        #     ''',
        #     llm_config = {
        #         "functions": [write_function],
        #         "config_list": config_list,
        #         "temperature": 0,
        #         "retry_wait_time": 30,
        #         "request_timeout": GLOBAL_TIMEOUT,
        #     },
        #     function_map={
        #         "write_content": write_content
        #     }
        # )

        copywriter = ChainlitAssistantAgent(
            name="Copywriter",
            system_message=f'''You are a Copywriter, you can use research function to collect latest information about a given topic, 
            and then use write_content function to write a very well written content;
            Reply TERMINATE when your task is done
            Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
            ''',
            llm_config = {
                "functions": [research_function, write_function],
                "config_list": config_list,
                "temperature": 0,
                "retry_wait_time": 30,
                "request_timeout": GLOBAL_TIMEOUT,
            },
            function_map={
                "research": research,
                "write_content": write_content
            }
        )

        graphic_designer = ChainlitAssistantAgent(
            name="Graphic_Designer",
            system_message=f'''As an expert in text-to-image AI models, you will utilize the 'generate_image' function to create an image based on the given prompt and iterate on the prompt. 
            Incorporating feedback from the Art Director until it achieves a perfect rating of 10/10.''',
            llm_config=llm_config_assistants,
            function_map={
                "image_review": review_image,
                "generate_image": generate_image
            }
        )

        art_director = ChainlitAssistantAgent(
            name="Art_Director",
            system_message=f'''You are the Art Director. 
            As an AI image critic, your task is to employ the 'image_review' function to evaluate the image generated by the Graphic Designer using the original prompt. 
            You will then offer feedback on how to enhance the prompt for better image generation.''',
            llm_config=llm_config_assistants,
            function_map={
                "image_review": review_image,
                "generate_image": generate_image
            }
        )

        user_proxy = ChainlitUserProxyAgent(
            name="User_Proxy",
            human_input_mode="TERMINATE",
            function_map={
                "research": research,
                "write_content": write_content,
                "image_review": review_image,
                "generate_image": generate_image
            }
        )

        cl.user_session.set(USER_PROXY_NAME, user_proxy)
        cl.user_session.set(PROJECT_MANAGER, project_manager)
        # cl.user_session.set(CREATIVE_DIRECTOR, creative_director)
        cl.user_session.set(CONTENT_RESEARCHER, content_researcher)
        cl.user_session.set(COPYWRITER, copywriter)
        cl.user_session.set(GRAPHIC_DESIGNER, graphic_designer)
        cl.user_session.set(ART_DIRECTOR, art_director)
        
        await cl.Message(content=WELCOME_MESSAGE, author="Chat").send()
        
    except Exception as e:
        print("Error: ", e)
        pass


@cl.on_message
async def run_conversation(message: cl.Message):
    try:
        print("Start logging...")
        autogen.ChatCompletion.start_logging()
        
        TASK = message.content
        print("Task: ", TASK)

        user_proxy = cl.user_session.get(USER_PROXY_NAME)
        project_manager = cl.user_session.get(PROJECT_MANAGER)
        # creative_director = cl.user_session.get(CREATIVE_DIRECTOR)
        content_researcher = cl.user_session.get(CONTENT_RESEARCHER)
        copywriter = cl.user_session.get(COPYWRITER)
        graphic_designer = cl.user_session.get(GRAPHIC_DESIGNER)
        art_director = cl.user_session.get(ART_DIRECTOR)
        
        groupchat = autogen.GroupChat(agents=[user_proxy, project_manager, content_researcher, copywriter, graphic_designer, art_director], messages=[], max_round=30)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)
        
        print("Group chat messages: ", len(groupchat.messages))
        
        if len(groupchat.messages) == 0:
            await cl.Message(content=f"""Starting agents on task: {TASK}...""").send()
            await cl.make_async(user_proxy.initiate_chat)( manager, message=TASK, )
        else:
            await cl.make_async(user_proxy.send)( manager, message=TASK, )

        # Display cost logs
        # logs = autogen.ChatCompletion.logged_history
        # print(logs)
        # conversation = next(iter(logs.values()))
        # cost = sum(conversation["cost"])

        # TOTAL_COST = float(cl.user_session.get("total_cost", 0))
        # cost += TOTAL_COST
        # cl.user_session.set("total_cost", cost)

        # cost_counter = cl.TaskList(name="Cost Counter", status="running")
        # await cost_counter.send()
        # cost_task = cl.Task(title=f"Total Cost in USD for this conversation: ${float(cl.user_session.get('total_cost', 0)):.2f}", status=cl.TaskStatus.DONE)
        # await cost_counter.add_task(cost_task)
        # await cost_counter.send()
        
    except Exception as e:
        print("Error: ", e)
        pass


