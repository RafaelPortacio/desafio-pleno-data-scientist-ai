"""
Conversational Respon            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }de - Responde perguntas conversacionais
"""

import json
import requests
from src.config.settings import OPENAI_BASE_URL, OPENAI_MODEL, get_openai_api_key
from src.config.prompts import CONVERSATIONAL_PROMPT

def conversational_responder_node(state):
    """
    Nó para respostas conversacionais (saudações, agradecimentos, etc.)
    """
    question = state.get("question", "")
    
    prompt = CONVERSATIONAL_PROMPT.format(question=question)
    
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
            final_response = result["choices"][0]["message"]["content"].strip()
        else:
            final_response = f"Olá! Sou o assistente de análise de dados da Prefeitura do Rio de Janeiro. Como posso ajudá-lo?"
        
        return {
            "final_response": final_response,
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": final_response
            }]
        }
        
    except Exception as e:
        fallback_response = "Olá! Sou o assistente de análise de dados da Prefeitura do Rio de Janeiro. Como posso ajudá-lo com informações sobre os serviços municipais?"
        return {
            "final_response": fallback_response,
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": fallback_response
            }]
        }