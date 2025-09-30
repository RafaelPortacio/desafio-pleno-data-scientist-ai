"""
Response Synthesizer Node - Converte dados de con            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }em resposta natural
"""

import json
import requests
from src.config.settings import OPENAI_BASE_URL, OPENAI_MODEL, get_openai_api_key
from src.config.prompts import RESPONSE_SYNTHESIZER_PROMPT

def response_synthesizer_node(state):
    """
    Nó sintetizador: converte dados em resposta natural
    """
    question = state.get("question", "")
    data_result = state.get("data_result", [])
    error = state.get("error")
    
    if error:
        # Se houve erro, gerar resposta de erro amigável
        prompt = f"""
        Houve um erro ao processar a consulta de dados para a pergunta: "{question}"
        
        Erro: {error}
        
        Forneça uma resposta amigável explicando que não foi possível obter os dados solicitados
        e sugira que o usuário reformule a pergunta ou tente novamente.
        """
    else:
        # Se temos dados, usar o template configurado
        if data_result and len(data_result) > 0:
            # Criar resumo dos dados
            data_summary = f"Dados encontrados ({len(data_result)} linhas):\n"
            if len(data_result) <= 10:
                data_summary += str(data_result)
            else:
                data_summary += str(data_result[:10])
                data_summary += f"\n... e mais {len(data_result) - 10} linhas"
        else:
            data_summary = "Nenhum resultado encontrado."
        
        prompt = RESPONSE_SYNTHESIZER_PROMPT.format(
            data_result=data_summary,
            question=question
        )
    
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
            final_response = f"Erro ao gerar resposta: {response.status_code}"
        
        return {
            "final_response": final_response,
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": final_response
            }]
        }
        
    except Exception as e:
        error_response = f"Erro ao sintetizar resposta: {str(e)}"
        return {
            "final_response": error_response,
            "messages": state.get("messages", []) + [{
                "role": "assistant", 
                "content": error_response
            }]
        }