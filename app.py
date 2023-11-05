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

from utilities.tools import generate_image, review_image, search, scrape
from utilities.chainlit_helpers import ChainlitAssistantAgent, ChainlitUserProxyAgent

# Load environment variables
load_dotenv()

# Load default config
PRO_GPT_MODEL = "gpt-4"
BASE_GPT_MODEL = "gpt-3.5-turbo-16k"
SUB_GPT_MODEL = "gpt-3.5-turbo"
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")
GLOBAL_TIMEOUT = os.getenv("GLOBAL_TIMEOUT")

USER_PROXY_NAME = "Query Agent"
DOMAIN_EXPERT = "Domain Expert"
CREATIVE_DIRECTOR = "Creative Director"
PROJECT_MANAGER = "Project Manager"
CONTENT_STRATEGIST = "Content Strategist"
CONTENT_RESEARCHER = "Content Researcher"
CONTENT_WRITER = "Content Writer"
WRITING_ASSISTANT = "Writing Assistant"
RESEARCH_ADMIN = "Research Admin"
EDITORIAL_ADMIN = "Editorial Admin"
ARTIST = "Artist"
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
    "config_list": config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": {"gpt-4", "gpt-3.5-turbo-16k"}}),
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

            cl.user_session.set(EDITORIAL_ADMIN, editorial_admin)

            editorial_team = autogen.GroupChat(
                agents=[editorial_admin, editor, writer, reviewer],
                messages=[],
                max_round=3)
            
            manager = autogen.GroupChatManager(groupchat=editorial_team, llm_config=llm_config)

            editorial_admin.initiate_chat(
                manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

            editorial_admin.stop_reply_at_receive(manager)
            editorial_admin.send(
                "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

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

            research_assistant = AssistantAgent(
                name="Research_Assistant",
                system_message=f'''
                Welcome, Research Assistant.
                Your task is to research the provided query extensively. 
                Produce a detailed report, ensuring you include technical specifics and reference all sources. Conclude your report with "TERMINATE".
                ''',
                llm_config=llm_config_researcher,
            )

            research_admin = UserProxyAgent(
                name="Research_Admin",
                code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
                is_termination_msg=lambda x: x.get("content", "") and x.get(
                    "content", "").rstrip().endswith("TERMINATE"),
                human_input_mode="NEVER",
                function_map={
                    "search": search,
                    "scrape": scrape,
                }
            )


            research_admin.initiate_chat(research_assistant, message=query)

            # Format for markdown (optional step)
            # formatted_report = format_for_markdown(research_admin.last_message()["content"])

            # Save the research report
            # save_to_file(formatted_report, "research_report")

            # set the receiver to be researcher, and get a summary of the research report
            research_admin.stop_reply_at_receive(research_assistant)
            research_admin.send(
                "Give me the research report that just generated again, return ONLY the report & reference links.", research_assistant)

            # return the last message the expert received
            return research_admin.last_message()["content"]

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
            You are the Project Manager. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
            Your primary responsibility is to oversee the entire project lifecycle, ensuring that all agents are effectively fulfilling their objectives and tasks on time.
            Based on the directives from the user task, coordinate with all involved agents, set clear milestones, and monitor progress. Ensure that user feedback is promptly incorporated, and any adjustments are made in real-time to align with the project's goals.
            Act as the central point of communication, facilitating collaboration between teams and ensuring that all deliverables are of the highest quality. Your expertise is crucial in ensuring that the project stays on track, meets deadlines, and achieves its objectives.
            Regularly review the project's status, address any challenges, and ensure that all stakeholders are kept informed of the project's progress.
            ''',
            llm_config = llm_config,
        )

        creative_director = ChainlitAssistantAgent(
            name="Creative_Director",
            system_message=f'''
            You are the Creative Director. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
            Your primary role is to guide the creative vision of the project, ensuring that all ideas are not only unique and compelling but also meet the highest standards of excellence and desirability.
            Drawing from the insights of user task, oversee the creative process, inspire innovation, and set the bar for what's possible. Challenge the team to think outside the box and push the boundaries of creativity.
            Review all creative outputs, provide constructive feedback, and ensure that every piece aligns with the brand's identity and resonates with the target audience. 
            Collaborate closely with all teams, fostering a culture of excellence, and ensuring that our creative solutions are both groundbreaking and aligned with the project's objectives.
            ''',
            llm_config = llm_config,
        )

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
            You must use the research function to provide a topic for the writing_assistant in order to get up to date information outside of your knowledge cutoff
            Your primary responsibility is to delve deep into understanding the challenges around mental health.
            Using the information from the user task, conduct thorough research to uncover insights related to the task.
            Your findings should shed light on mental health strategies, meditation techniques, mindfulness practices, and other therapeutic methods.
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

        content_writer = ChainlitAssistantAgent(
            name="Content_Copywriter",
            system_message=f'''
            You are the Lead Copywriter.
            Your primary role is to craft compelling narratives and messages that align with the organisation's vision to break the stigma of mental health in Asia, so that people can get the help they need.
            Based on the research gathered from the Content Researcher, create engaging content, from catchy headlines to in-depth articles.
            Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
            ''',
            llm_config = {
                "functions": [write_function],
                "config_list": config_list,
                "temperature": 0,
                "retry_wait_time": 30,
                "request_timeout": GLOBAL_TIMEOUT,
            },
            function_map={
                "write_content": write_content
            }
        )

        writing_assistant = ChainlitAssistantAgent(
            name="Writing_Assistant",
            system_message=f'''You are a writing assistant, you can use research function to collect latest information about a given topic, 
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

        artist = ChainlitAssistantAgent(
            name="Artist",
            system_message="As an expert in text-to-image AI models, you will utilize the 'generate_image' function to create an image based on the given prompt and iterate on the prompt, incorporating feedback until it achieves a perfect rating of 10/10.",
            llm_config=llm_config_assistants,
            function_map={
                "image_review": review_image,
                "generate_image": generate_image
            }
        )

        art_director = ChainlitAssistantAgent(
            name="Art_Director",
            system_message="In the role of an AI image critic, your task is to employ the 'image_review' function to evaluate the image generated by the 'Artist' using the original prompt. You will then offer feedback on how to enhance the prompt for better image generation.",
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
        cl.user_session.set(CREATIVE_DIRECTOR, creative_director)
        cl.user_session.set(CONTENT_RESEARCHER, content_researcher)
        cl.user_session.set(CONTENT_WRITER, content_writer)
        cl.user_session.set(WRITING_ASSISTANT, writing_assistant)
        cl.user_session.set(ARTIST, artist)
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
        creative_director = cl.user_session.get(CREATIVE_DIRECTOR)
        content_researcher = cl.user_session.get(CONTENT_RESEARCHER)
        content_writer = cl.user_session.get(CONTENT_WRITER)
        writing_assistant = cl.user_session.get(WRITING_ASSISTANT)
        artist = cl.user_session.get(ARTIST)
        art_director = cl.user_session.get(ART_DIRECTOR)
        
        groupchat = autogen.GroupChat(agents=[user_proxy, project_manager, creative_director, content_researcher, content_writer, writing_assistant, artist, art_director], messages=[], max_round=20)
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


