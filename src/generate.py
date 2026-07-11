import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def generate_answer(query, contexts):
    """Generates an answer to the query using the retrieved contexts, enforcing citations and guardrails."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    client = genai.Client(api_key=api_key)
    
    # 1. Format the context chunks for the prompt
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += f"--- Source {idx+1}: {ctx['document_name']} (Page {ctx['page_number']}) ---\n"
        context_str += f"{ctx['text']}\n\n"
        
    # 2. Design the prompt with strict rules and guardrails
    system_instruction = (
        "You are an expert academic AI assistant for StudyBuddy EdTech. Your job is to answer "
        "questions about study materials using ONLY the provided sources. Follow these rules:\n\n"
        "1. STRICT EVIDENCE-BASED ANSWERING: Answer the question using ONLY the provided sources. Do not "
        "bring in external knowledge. If the provided sources do not contain the answer, or if the "
        "evidence is weak or irrelevant, you MUST reply exactly with: "
        "'I do not know the answer based on the provided corpus. (Refused: Out of corpus or insufficient evidence)'\n\n"
        "2. INLINE CITATIONS: For every claim, fact, or assertion you make, you must cite the source inline. "
        "Use the exact format: [Document_Name, p. X] (e.g., [Attention_Is_All_You_Need, p. 5]). "
        "Citations must point to the specific document and page number where the information was found. "
        "You can include multiple citations if you synthesize facts from different pages.\n\n"
        "3. DIRECT QUOTATION: Where appropriate, quote the source text directly to support your answer, "
        "along with its citation.\n\n"
        "4. OUT-OF-CORPUS GUARDRAILS: If the question is about generic topics (e.g., 'What is the capital of France?', "
        "'How do I write a binary search?'), refuse to answer immediately using the standard refusal message above.\n\n"
        "5. OBJECTIVE TONE: Maintain a professional, neutral, academic tone. Be direct and clear."
    )
    
    prompt = f"""
{system_instruction}

Here are the retrieved sources:
{context_str}

User Question: {query}
Answer:
"""
    
    # 3. Call the generation model
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error during generation: {e}"

if __name__ == "__main__":
    # Test generation with mock context
    mock_contexts = [
        {
            "document_name": "Attention_Is_All_You_Need",
            "page_number": 3,
            "text": "The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours."
        }
    ]
    test_query = "How long was the transformer trained to reach state of the art?"
    print("Testing Generation...")
    print(generate_answer(test_query, mock_contexts))
