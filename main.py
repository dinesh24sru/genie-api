from fastapi import FastAPI, HTTPException 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()  
 
openai.api_key =  os.environ.get("OPENAI_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lively-forest-09162750f.5.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel): 
    role: str
    content: str
 
@app.get("/")
async def helloapp():
    return {"message": "Hello App"}


@app.on_event("startup")
async def startup_event():
    global index, chat_engine
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.3,
        system_prompt= """ You are a specialized expert in QMS Standards for the American Petroleum Institute (API) and the International Organization for Standardization (ISO) working for Vegas Consulting Services (VegasCG)
                        Your role is to provide accurate, technical answers strictly based on the provided data. Respond only to questions about monogram standards and avoid any non-technical inquiries.
                        If a question falls outside your expertise, reply that you are designed to address questions related to API and ISO standards only.
                        Ensure your answers are fact-based, precise, and free of unsupported claims or hallucinations. You must never answer any non-technical questions
                        Give yourself room to think by extracting relevant passages from the context before answering the query.
                        Don't return the thinking, only return the answer. Make sure your answers are as explanatory as possible."""
                    ) 

    index = VectorStoreIndex.from_documents(docs)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )


async def enhance_response(response: str, query: str) -> str:
    """Enhance the response by providing additional context or clarification."""
    prompt =  (
        f"Enhance the following response based on the query:\n\n" 
        f"Query: {query}\n"
        f"Response: {response}\n\n"
        """Provide a more detailed and accurate response. Use lists, bold texts, italics, bullets, points, etc. to visualise attractively. Return the response in markdown format.
        Don't display anything like 'Here is an enhanced response'. The answer must be structured and visually attractive.
        You are a specialized chatbot designed to answer only technical questions related to VegasCG and QMS Standards for the American Petroleum Institute (API) and the International Organization for Standardization (ISO).
        Follow these rules:
        1. Only provide answers to questions that are strictly technical and related to VegasCG or QMS Standards for API and ISO. 
        2. For every technical question, provide a clear, concise answer and an explanation. Include a reference for further reading but summarize the answer without directing the user to the reference. 
        3. If a non-technical question is asked or a question not related to VegasCG or QMS Standards, respond with: 'Sorry, I am defined only to answer queries related to monogram standards.'
 
        Please make sure to follow these rules strictly. """
    )

    try:
        openai_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert providing enhanced answers based on given queries and responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        enhanced_content = openai_response.choices[0].message.content
        return enhanced_content.strip()
    except Exception as e:
        print(f"Error enhancing response: {e}")
        return response

@app.options("/chat")
async def options_chat_endpoint():
    return {}

@app.post("/chat")
async def chat(message: Message):
    if message.role != "user":
        raise HTTPException(status_code=400, detail="Invalid role")

    # Generate initial response from the chatbot
    response_stream = chat_engine.stream_chat(message.content)
    response_chunks = [chunk for chunk in response_stream.response_gen]
    response = "".join(response_chunks)

    # Enhance the response for better clarity
    enhanced_response = await enhance_response(response, message.content)
    return {"role": "assistant", "content": enhanced_response}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


