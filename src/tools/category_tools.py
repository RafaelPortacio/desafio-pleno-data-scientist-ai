"""
Tools para busca por similaridade em colunas categóricas
Utiliza ChromaDB e embeddings OpenAI para encontrar valores similares
"""

import os
from dotenv import load_dotenv
import requests
import chromadb
from typing import List, Optional, Dict
from langchain_core.tools import tool
from src.config.settings import (
    OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTIONS, SIMILARITY_THRESHOLD, get_openai_api_key
)

load_dotenv()

class OpenAIClient:
    """Custom OpenAI client using requests"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = OPENAI_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_embeddings(self, texts: List[str], model: str = OPENAI_EMBEDDING_MODEL) -> List[List[float]]:
        """Create embeddings using OpenAI API"""
        url = f"{self.base_url}/embeddings"
        data = {
            "model": model,
            "input": texts,
            "encoding_format": "float"
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]

class CategorySearchTools:
    """Tools para busca por similaridade em categorias"""
    
    def __init__(self):
        self.openai_client = OpenAIClient(api_key=get_openai_api_key())
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.similarity_threshold = SIMILARITY_THRESHOLD
    
    def _search_similar(self, collection_name: str, query: str, n_results: int = 5) -> List[Dict]:
        """Busca valores similares em uma coleção"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Verificar se a coleção tem dados
            if collection.count() == 0:
                print(f"⚠️ Coleção {collection_name} está vazia")
                return []
            
            # Criar embedding da query
            embeddings = self.openai_client.create_embeddings(
                texts=[query],
                model=OPENAI_EMBEDDING_MODEL
            )
            query_embedding = embeddings[0]
            
            # Buscar
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Filtrar por threshold de similaridade
            similar_values = []
            for doc, distance in zip(results['documents'][0], results['distances'][0]):
                similarity = 1 - distance  # Converter distância cosseno para similaridade
                if similarity >= self.similarity_threshold:
                    similar_values.append({
                        "value": doc,
                        "similarity": similarity
                    })
            
            return similar_values
            
        except Exception as e:
            print(f"⚠️ Erro na busca em {collection_name}: {e}")
            return []

# Instanciar as tools
category_tools = CategorySearchTools()

@tool
def get_nome_unidade_organizacional(query: str) -> str:
    """
    Busca nomes de unidades organizacionais similares ao termo fornecido.
    
    Args:
        query: Termo de busca para encontrar unidades organizacionais similares
        
    Returns:
        String com os valores similares encontrados ou mensagem de não encontrado
    """
    results = category_tools._search_similar(CHROMA_COLLECTIONS["nome_unidade_organizacional"], query)
    
    if not results:
        return f"Busca por similaridade não disponível para '{query}'. Use padrões LIKE na consulta SQL."
    
    similar_values = [f"'{r['value']}' (sim: {r['similarity']:.3f})" for r in results]
    return f"Unidades organizacionais similares a '{query}': {', '.join(similar_values)}"

@tool  
def get_id_unidade_organizacional_mae(query: str) -> str:
    """
    Busca unidades organizacionais mãe similares ao termo fornecido.
    
    Args:
        query: Termo de busca para encontrar unidades mãe similares
        
    Returns:
        String com os valores similares encontrados ou mensagem de não encontrado
    """
    results = category_tools._search_similar(CHROMA_COLLECTIONS["id_unidade_organizacional_mae"], query)
    
    if not results:
        return f"Busca por similaridade não disponível para '{query}'. Use padrões LIKE na consulta SQL."
    
    similar_values = [f"'{r['value']}' (sim: {r['similarity']:.3f})" for r in results]
    return f"Unidades mãe similares a '{query}': {', '.join(similar_values)}"

@tool
def get_tipo(query: str) -> str:
    """
    Busca tipos de chamados similares ao termo fornecido.
    
    Args:
        query: Termo de busca para encontrar tipos de chamados similares
        
    Returns:
        String com os valores similares encontrados ou mensagem de não encontrado
    """
    results = category_tools._search_similar(CHROMA_COLLECTIONS["tipo"], query)
    
    if not results:
        return f"Busca por similaridade não disponível para '{query}'. Use padrões LIKE na consulta SQL."
    
    similar_values = [f"'{r['value']}' (sim: {r['similarity']:.3f})" for r in results]
    return f"Tipos similares a '{query}': {', '.join(similar_values)}"

@tool
def get_subtipo(query: str) -> str:
    """
    Busca subtipos de chamados similares ao termo fornecido.
    
    Args:
        query: Termo de busca para encontrar subtipos de chamados similares
        
    Returns:
        String com os valores similares encontrados ou mensagem de não encontrado
    """
    results = category_tools._search_similar(CHROMA_COLLECTIONS["subtipo"], query)
    
    if not results:
        return f"Busca por similaridade não disponível para '{query}'. Use padrões LIKE na consulta SQL."
    
    similar_values = [f"'{r['value']}' (sim: {r['similarity']:.3f})" for r in results]
    return f"Subtipos similares a '{query}': {', '.join(similar_values)}"

# Lista de todas as tools disponíveis
CATEGORY_TOOLS = [
    get_nome_unidade_organizacional,
    get_id_unidade_organizacional_mae,
    get_tipo,
    get_subtipo
]