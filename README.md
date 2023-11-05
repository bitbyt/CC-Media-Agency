# Calm Collective Media Agency: Multi Agent Team with Autogen

## 📖 Introduction

An exploration into the possibilty of a fully autonomous Media Agency workforce using the [AutoGen library](https://github.com/microsoft/autogen). Instead of relying on a single agent to handle tasks, multiple specialized agents work together, each bringing its expertise to the table.

## 🧑🏻‍💻 AI Roles

The agents involved in the collaboration include:

1. **Project Manager**
    - Plans and manages the task provided.
    - Colloborates with the team to ensure smooth delivery of project.
2. **Creative Director**
    - Manages the creative output of the team.
3. **Art Director**
    - Reviews the output of the artist and provide feedback to improve on the artwork.
4. **Artist**
    - Create any artwork required for the project.
5. **Researcher**
    - Conducts research on a given subject, provides insight for the team.
    - Create research reports.
6. **Copywriter**
    - Create articles, short form or long form content.
7. **Domain Expert**
    - Provides domain knowledge of Calm Collective, past talks, articles and mental health resources.
8. **User Proxy**
    - Acts as an intermediary between the human user and the agents.

## 🛠️ Setup & Configuration

1. Ensure required libraries are installed:
```
pip install pyautogen
```

2. Set up the OpenAI configuration list by either providing an environment variable `OAI_CONFIG_LIST` or specifying a file path.
```
[
    {
        "model": "gpt-3.5-turbo", #or whatever model you prefer
        "api_key": "INSERT_HERE"
    }
]
```

3. Setup api keys in .env:
```
OPENAI_API_KEY="XXX"
SERP_API_KEY="XXX"
BROWSERLESS_API_KEY="XXX"
REPLICATE_API_TOKEN="XXX"
CHAINLIT_API_KEY="XXX"
```

4. Launch in CLI:
```
python3 main.py
```

## 📈 Roadmap

1. Refine workflow and data pass through to agents
2. Reduce unnecessaery back and forth
3. Save files to local folder
4. Implement other agents, see commented out agents
6. Create and train fine-tuned agents for each domain specific task

## 📝 License 

MIT License. See [LICENSE](https://opensource.org/license/mit/) for more information.