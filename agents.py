# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

#Pipenv only auto-loads .env when you use pipenv shell or pipenv run — if you're running python backend.py directly inside the shell without pipenv run, it won't load. That's why Groq worked (probably set as system env var) but Google didn't.

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="Google":
        llm=ChatGoogleGenerativeAI(model=llm_id, google_api_key=GOOGLE_API_KEY)

    tools=[search_tool] if allow_search else []
    
    agent=create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    query_text = " ".join(query) if isinstance(query, list) else query
    
    state={"messages": [("user", query_text)]}  # we need to pass the query in the format of messages to the agent, as it expects a list of messages with role and content, here we are only passing one user message, but in future we can easily extend this to pass multiple messages and also ai messages if we want to continue the conversation with context
    response=agent.invoke(state)  # this contains all meta data and tool calls, messages human message and ai messages 
    messages=response.get("messages") # first we filter out all messages from the response

    # ✅ Handle both Groq (returns plain string) and Gemini (returns list of dicts with type and text keys) response formats
    ai_messages = []
    for message in messages:
        if isinstance(message, AIMessage):
            if isinstance(message.content, str) and message.content:
                # Groq returns plain string content
                ai_messages.append(message.content)
            elif isinstance(message.content, list):
                # Gemini returns a list of dicts, we extract only the text blocks
                text = " ".join(
                    block["text"] for block in message.content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text:
                    ai_messages.append(text)

    return ai_messages[-1] if ai_messages else "No response generated." # return the last ai message which is the final response to the user, as there could be multiple ai messages in the thought process and tool calls