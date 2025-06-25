import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from fastapi import FastAPI, File, UploadFile, Form
from google.colab import userdata
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


# Load API keys
groq_key = userdata.get('groqdummy')
tavily_key = userdata.get('tavily')

# Initialize Groq client
# client = Groq(api_key=groq_key) # Note: This client object is not directly used in the agent, but the key is used by init_chat_model

# Initialize tools
search = TavilySearchResults(max_results=1, tavily_api_key=tavily_key)

# Initialize RAG components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = WebBaseLoader("https://www.atomcamp.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "Atomcamp",
    "Information about Atomcamp . For any questions about Atomcamp, you must use this tool!",
)

tools = [search, retriever_tool]

# Initialize the language model and agent
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq", api_key=groq_key)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Define FastAPI app
app = FastAPI(title = "Our FastAPI app")

@app.get("/")
def home():
    return "Welcome to Our FastAPI Endpoints"

@app.get("/data")
def data():
    return {"data": "Here is your data bla bla"}

@app.get("/agent")
def agent_endpoint(input_string: str):
    response = agent_executor.invoke({"input": input_string})
    return response.get('output', 'Could not process the request') # Return the output key from the response


# if __name__ == "__main__":
#     # ngrok setup is typically for local development/testing,
#     # for cloud deployment you would use the cloud provider's
#     # built-in methods for exposing the application.
#     # ngrok_tunnel = ngrok.connect(8000)
#     # print("Public URL:", ngrok_tunnel.public_url)
#     # nest_asyncio.apply() # This is also for interactive environments like Colab

#     uvicorn.run(app, host="0.0.0.0", port=8000) # Listen on all interfaces for cloud deployment
