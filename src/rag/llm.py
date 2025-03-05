# src/rag/llm.py
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query: str, context: str, model: str) -> str:
    """Generate an answer using OpenAI API"""
    prompt = f"""You are an assistant that answers questions based on the provided context. 
If the answer cannot be determined from the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating with OpenAI: {str(e)}")
        return f"OpenAi encountered an error while generating a response: {str(e)}"