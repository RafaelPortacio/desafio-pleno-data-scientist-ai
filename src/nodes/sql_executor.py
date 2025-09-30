"""
SQL Executor Node - Executa consultas SQL no BigQuery
"""

from google.cloud import bigquery
from src.config.settings import get_google_credentials

def sql_executor_node(state):
    """
    Nó executor de SQL: executa a consulta no BigQuery
    """
    sql_query = state.get("sql_query")
    
    if not sql_query or sql_query.strip() == "":
        print("❌ ERRO: SQL query está vazia!")
        return {
            "error": "SQL query vazia ou inválida",
            "data_result": []
        }
    
    try:
        # Inicializar cliente BigQuery usando service account file
        credentials_path = get_google_credentials()  # Valida que existe
        bigquery_client = bigquery.Client.from_service_account_json(credentials_path)
        
        # Executar consulta
        print(f"🔍 Executando SQL:\n{sql_query}")
        results = bigquery_client.query(sql_query)
        
        # Converter para lista de dicionários
        df = results.to_dataframe().to_dict(orient="records")
        print(f"✅ Consulta executada com sucesso. {len(df)} linhas retornadas.")
        
        return {
            "data_result": df,
            "messages": state.get("messages", []) + [{
                "role": "system", 
                "content": f"Consulta executada com sucesso. {len(df)} linhas retornadas."
            }]
        }
            
    except Exception as e:
        error_msg = f"Erro ao executar SQL: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "error": error_msg,
            "data_result": [],
            "messages": state.get("messages", []) + [{
                "role": "system", 
                "content": error_msg
            }]
        }