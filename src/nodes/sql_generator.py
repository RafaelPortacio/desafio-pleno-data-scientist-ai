"""
SQL Generator Node - Gera consultas SQL baseadas na pergunta do usuário
"""

import json
import requests
from src.config.settings import OPENAI_BASE_URL, OPENAI_MODEL, get_openai_api_key, load_schema
from src.config.prompts import SQL_GENERATOR_SYSTEM_PROMPT, TOOL_CONTEXT_PROMPT
from src.tools.category_tools import (
    get_nome_unidade_organizacional,
    get_id_unidade_organizacional_mae,
    get_tipo,
    get_subtipo
)

def sql_generator_node(state):
    """
    Nó gerador de SQL que cria consultas baseadas na pergunta
    """
    question = state.get("question", "")
    messages = state.get("messages", [])
    
    # Carrega schemas
    schema_chamado = load_schema("chamado")
    schema_bairro = load_schema("bairro")
    
    # Define as tools disponíveis
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_nome_unidade_organizacional",
                "description": "Busca nomes de unidades organizacionais similares ao termo fornecido",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo para buscar unidades organizacionais similares"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_id_unidade_organizacional_mae",
                "description": "Busca IDs de unidades organizacionais mãe similares ao termo fornecido",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo para buscar unidades mãe similares"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_tipo",
                "description": "Busca tipos de chamados similares ao termo fornecido",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo para buscar tipos similares"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_subtipo",
                "description": "Busca subtipos de chamados similares ao termo fornecido",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo para buscar subtipos similares"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Monta o prompt inicial
    agent_scratchpad = ""
    if messages:
        agent_scratchpad = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    system_prompt = SQL_GENERATOR_SYSTEM_PROMPT.format(
        schema_chamado=schema_chamado,
        schema_bairro=schema_bairro,
        agent_scratchpad=agent_scratchpad
    )
    
    try:
        api_key = get_openai_api_key()
        
        # Primeira chamada para identificar se precisa usar tools
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": initial_messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 1
            }
        )
        
        if response.status_code != 200:
            return {"sql_query": None, "error": f"Erro na API: {response.status_code}"}
        
        result = response.json()
        message = result["choices"][0]["message"]
        
        # Verifica se há tool calls
        tool_context = ""
        if "tool_calls" in message and message["tool_calls"]:
            tool_results = []
            
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                # Executa a tool correspondente
                if function_name == "get_nome_unidade_organizacional":
                    result_tool = get_nome_unidade_organizacional(function_args["query"])
                elif function_name == "get_id_unidade_organizacional_mae":
                    result_tool = get_id_unidade_organizacional_mae(function_args["query"])
                elif function_name == "get_tipo":
                    result_tool = get_tipo(function_args["query"])
                elif function_name == "get_subtipo":
                    result_tool = get_subtipo(function_args["query"])
                else:
                    result_tool = "Tool não encontrada"
                
                tool_results.append(f"{function_name}('{function_args['query']}'): {result_tool}")
            
            tool_context = "\n".join(tool_results)
            
            # Segunda chamada com contexto das tools
            tool_prompt = TOOL_CONTEXT_PROMPT.format(
                tool_context=tool_context,
                schema_chamado=schema_chamado,
                schema_bairro=schema_bairro
            )
            
            final_messages = [
                {"role": "system", "content": tool_prompt},
                {"role": "user", "content": question}
            ]
            
            final_response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": final_messages,
                    "temperature": 1
                }
            )
            
            if final_response.status_code == 200:
                final_result = final_response.json()
                sql_content = final_result["choices"][0]["message"]["content"]
            else:
                sql_content = message.get("content", "")
        else:
            sql_content = message.get("content", "")
        
        # Extrai apenas o SQL do conteúdo
        sql_query = extract_sql_from_content(sql_content)
        
        return {
            "sql_query": sql_query,
            "tool_context": tool_context,
            "messages": messages + [{"role": "assistant", "content": sql_content}]
        }
        
    except Exception as e:
        return {"sql_query": None, "error": f"Erro ao gerar SQL: {str(e)}"}

def extract_sql_from_content(content):
    """
    Extrai o SQL do conteúdo retornado pelo modelo
    """
    if not content:
        return None
    
    # Procura por SQL: ou sql: ou similar
    lines = content.split('\n')
    sql_lines = []
    capturing = False
    
    for line in lines:
        if line.strip().upper().startswith('SQL:'):
            capturing = True
            sql_part = line.split(':', 1)
            if len(sql_part) > 1 and sql_part[1].strip():
                sql_lines.append(sql_part[1].strip())
            continue
        elif capturing and line.strip():
            if line.strip().upper().startswith(('REASONING:', 'EXPLANATION:', 'NOTE:')):
                break
            sql_lines.append(line)
        elif capturing and not line.strip():
            # Linha vazia pode ser parte do SQL
            sql_lines.append(line)
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    # Fallback: procura por SELECT
    if 'SELECT' in content.upper():
        # Tenta extrair tudo que parece SQL
        start_idx = content.upper().find('SELECT')
        if start_idx != -1:
            sql_part = content[start_idx:]
            # Remove explicações posteriores
            for stop_word in ['REASONING:', 'EXPLANATION:', 'NOTE:']:
                if stop_word in sql_part.upper():
                    sql_part = sql_part[:sql_part.upper().find(stop_word)]
            return sql_part.strip()
    
    return content.strip()