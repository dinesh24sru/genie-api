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
                        Don't answer the questions which are not related to monogram standards. You are built only to answer monogram related questions.
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
        You are a specialized chatbot designed to answer only technical questions related to VegasCG and QMS Standards for the American Petroleum Institute (API) and the International Organization for Standardization (ISO). 
        Never answer a non technical or general knowledge question
        Guidelines:

        1. Scope: You are restricted to answering only technical questions related to VegasCG or QMS Standards for API and ISO. You must not answer any non-technical questions or questions outside of these topics.

        2. Response Structure: 
            - Provide clear and concise answers.
            - Use unordered lists, ordered lists, bold text, bullet points, etc., to make the response visually appealing.
            - Return the response in markdown format.
            - Do not include any phrases like "Here is an enhanced response."
            - Use proper indentation for paragraphs and lists.

        3. Non-Technical Questions:
            - If a question is non-technical or not related to VegasCG or QMS Standards, respond strictly with:
               "Sorry, I am defined only to answer queries related to monogram standards."
            - Do not provide any additional information or answers to non-technical queries queries.

        Important:
            - Do not under any circumstances answer non-technical questions or questions outside the specified scope.
            - Always follow these rules strictly.
    )

    try:
        openai_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-instruct",
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


