# from langchain_groq import ChatGroq
# from langchain_community.tools.tavily_search import TavilySearchResults

# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages.ai import AIMessage

# from app.config.settings import settings

# def get_response_from_ai_agents(llm_id , query , allow_search ,system_prompt):

#     llm = ChatGroq(model=llm_id)

#     tools = [TavilySearchResults(max_results=2)] if allow_search else []

#     agent = create_react_agent(
#         model=llm,
#         tools=tools,
#         state_modifier=system_prompt
#     )

#     state = {"messages" : query}

#     response = agent.invoke(state)

#     messages = response.get("messages")

#     ai_messages = [message.content for message in messages if isinstance(message,AIMessage)]

#     return ai_messages[-1]





from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from app.config.settings import settings

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id, api_key=settings.GROQ_API_KEY)
    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create agent WITHOUT state_modifier
    agent = create_react_agent(model=llm, tools=tools)

    # Prepend system prompt manually
    state = {
        "messages": [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": m} for m in query]
    }

    response = agent.invoke(state)

    messages = response.get("messages", [])
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response from agent"



