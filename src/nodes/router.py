"""
Router Node - Determina se a pergunta requer consulta a dados ou resposta conversacional
"""

import json
import requests
from src.config.settings import OPENAI_BASE_URL, OPENAI_MODEL, get_openai_api_key
from src.config.prompts import ROUTER_PROMPT

def router_node(state):
    """
    Nó roteador que decide se a pergunta requer consulta a dados ou é conversacional
    """
    question = state.get("question", "")
    
    prompt = ROUTER_PROMPT.format(question=question)
    
    try:
        api_key = get_openai_api_key()
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            decision = result["choices"][0]["message"]["content"].strip().lower()
            
            if "data_query" in decision:
                return {"route": "data_query"}
            else:
                return {"route": "conversational"}
        else:
            print(f"Erro na API OpenAI: {response.status_code}")
            return {"route": "conversational"}
            
    except Exception as e:
        print(f"Erro no router: {e}")
        return {"route": "conversational"}