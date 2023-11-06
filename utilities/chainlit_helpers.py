import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent

import json

from typing import Dict, Optional, Union
from datetime import datetime

import chainlit as cl


logs_filename = f"logs/conversations_{datetime.now().timestamp()}.json"

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

def save_logs(logs_filename=logs_filename):
    logs = autogen.ChatCompletion.logged_history
    json.dump(logs, open(logs_filename, "w"), 
                indent=4)
    return logs

async def message_helper(data):
    content = ""
    if type(data["message"]) is str:
        message = data["message"]
        content = f'*Sending message to "{data["recipient"]}":*\n\n{message}'
    else:
        if data["message"]["role"] == 'function':
            message = data["message"]["content"]
            content = f'*Sending message to "{data["recipient"]}":*\n\n ** Response from calling function "{data["message"]["name"]}" ** \n\n{message}'
        print('message: ', data["message"]["content"])
    if content:
        await cl.Message(
            content=content,
            author=data["author"],
        ).send()

class ChainlitAssistantAgent(AssistantAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        message_content = {"author": self.name, "recipient": recipient.name, "message": message}
        cl.run_sync(
            message_helper(message_content)
        )
        super(ChainlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitUserProxyAgent(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Please give feedback to"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="continue", label="✅ Continue"
                        ),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        cl.Action( 
                            name="exit",
                            value="exit", 
                            label="🔚 Exit Conversation" 
                        ),
                    ],
                )
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=60))

        return reply["content"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                author=self.name,
            ).send()
        )
        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )