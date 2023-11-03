# def save_to_file(content, filename):
#     if not filename.endswith(".md"):
#         filename += ".md"
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write(content) 

# def format_for_markdown(content):
#     formatted_content = "# Research Report\n\n"  # Markdown header
#     formatted_content += content.replace("\n", "\n- ")  # Replace newlines with bullet points
#     return formatted_content

# Define research function
# def research(query):
#     llm_config_researcher = {
#         "functions": [
#             {
#                 "name": "search",
#                 "description": "Google search for relevant information",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "Google search query",
#                         }
#                     },
#                     "required": ["query"],
#                 },
#             },
#             {
#                 "name": "scrape",
#                 "description": "Scraping website content based on url",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "url": {
#                             "type": "string",
#                             "description": "Website url to scrape",
#                         }
#                     },
#                     "required": ["url"],
#                 },
#             },
#         ],
#         "config_list": config_list,
#         "temperature": 0,
#         "retry_wait_time": 60,
#         "request_timeout": 60,
#     }

#     research_assistant = AssistantAgent(
#         name="Research_Assistant",
#         system_message=f'''
#         Welcome, Research Assistant.
#         Your task is to research the provided query extensively. 
#         Produce a detailed report, ensuring you include technical specifics and reference all sources. Conclude your report with "TERMINATE".
#         ''',
#         llm_config=llm_config_researcher,
#     )

#     research_admin = ChainlitUserProxyAgent(
#         name="Research_Admin",
#         code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
#         is_termination_msg=lambda x: x.get("content", "") and x.get(
#             "content", "").rstrip().endswith("TERMINATE"),
#         human_input_mode="NEVER",
#         function_map={
#             "search": search,
#             "scrape": scrape,
#         }
#     )

#     cl.user_session.set(RESEARCH_ADMIN, research_admin)

#     research_admin.initiate_chat(research_assistant, message=query)

#     # Format for markdown (optional step)
#     # formatted_report = format_for_markdown(research_admin.last_message()["content"])

#     # Save the research report
#     # save_to_file(formatted_report, "research_report")

#     # set the receiver to be researcher, and get a summary of the research report
#     research_admin.stop_reply_at_receive(research_assistant)
#     research_admin.send(
#         "Give me the research report that just generated again, return ONLY the report & reference links.", research_assistant)

#     # return the last message the expert received
#     return research_admin.last_message()["content"]