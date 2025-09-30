"""
Configurações gerais do agen# Model Configuration
TEMPERATURE = 1
"""

import os

# OpenAI Configuration
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTIONS = {
    "tipo": "tipo_collection",
    "subtipo": "subtipo_collection", 
    "nome_unidade_organizacional": "unidade_organizacional_collection",
    "id_unidade_organizacional_mae": "unidade_mae_collection"
}

# Similarity Search Configuration
SIMILARITY_THRESHOLD = 0.3
SIMILARITY_TOP_K = 3

# Schema Files
SCHEMA_FILES = {
    "chamado": "static/schemas/schema_chamado.txt",
    "bairro": "static/schemas/schema_bairro.txt"
}

# Model Configuration
TEMPERATURE = 1

# Thread Pool Configuration  
THREAD_POOL_MAX_WORKERS = 3
BATCH_SIZE = 1000

# Warning Suppression
SUPPRESS_WARNINGS = True

def load_schema(schema_name):
    """Carrega o conteúdo de um arquivo de schema"""
    file_path = SCHEMA_FILES.get(schema_name)
    if not file_path:
        return f"Schema {schema_name} não encontrado"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Arquivo de schema {file_path} não encontrado"
    except Exception as e:
        return f"Erro ao carregar schema {schema_name}: {str(e)}"

def get_openai_api_key():
    """Obtém a chave da API do OpenAI das variáveis de ambiente"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
    return api_key

def get_google_credentials():
    """Obtém as credenciais do Google das variáveis de ambiente"""
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS não encontrada nas variáveis de ambiente")
    return credentials_path