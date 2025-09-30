"""
Script para inicializar ChromaDBs com dados categóricos do BigQuery
Cria embeddings para busca por similaridade nas colunas categóricas
"""

import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
import requests
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from src.config.settings import (
    OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTIONS, get_openai_api_key, get_google_credentials,
    THREAD_POOL_MAX_WORKERS, BATCH_SIZE
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

class CategoryEmbeddingsInitializer:
    """Inicializa embeddings para colunas categóricas"""
    
    def __init__(self):
        credentials_path = get_google_credentials()  # Valida credenciais
        self.bq_client = bigquery.Client.from_service_account_json(credentials_path)
        self.openai_client = OpenAIClient(api_key=get_openai_api_key())
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    
    def extract_unique_values(self, column_name: str) -> List[str]:
        """Extrai valores únicos de uma coluna categórica"""
        query = f"""
        SELECT DISTINCT {column_name}
        FROM `datario.adm_central_atendimento_1746.chamado`
        WHERE {column_name} IS NOT NULL
        AND {column_name} != ''
        ORDER BY {column_name}
        """
        
        print(f"📥 Extraindo valores únicos para: {column_name}")
        results = self.bq_client.query(query)
        values = [row[0] for row in results if row[0] and str(row[0]).strip()]
        print(f"✅ {len(values)} valores únicos encontrados para {column_name}")
        return values
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Cria embeddings para um lote de textos com retry e fallback"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not texts:
                    return []
                
                # Filtrar textos válidos
                valid_texts = [str(text).strip() for text in texts if text and str(text).strip()]
                if not valid_texts:
                    return []
                
                embeddings = self.openai_client.create_embeddings(valid_texts, OPENAI_EMBEDDING_MODEL)
                return embeddings
                
            except Exception as e:
                print(f"⚠️ Erro no lote (tentativa {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                else:
                    print(f"❌ Falhou após {max_retries} tentativas para o lote")
                    return []
    
    def process_batch_parallel(self, batch_data):
        """Processa um lote de dados em paralelo"""
        batch_id, texts = batch_data
        start_time = time.time()
        
        embeddings = self.create_embeddings_batch(texts)
        
        elapsed_time = time.time() - start_time
        embeddings_per_second = len(texts) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"📦 Lote {batch_id}: {len(texts)} embeddings em {elapsed_time:.2f}s ({embeddings_per_second:.1f} emb/s)")
        
        return batch_id, texts, embeddings
    
    def initialize_collection(self, collection_name: str, column_name: str):
        """Inicializa uma coleção ChromaDB com embeddings paralelos"""
        print(f"\n🚀 Iniciando inicialização da coleção: {collection_name}")
        
        # Extrair valores únicos
        unique_values = self.extract_unique_values(column_name)
        
        if not unique_values:
            print(f"⚠️ Nenhum valor encontrado para {column_name}")
            return
        
        # Verificar se a coleção já existe e deletar
        try:
            existing_collection = self.chroma_client.get_collection(collection_name)
            self.chroma_client.delete_collection(collection_name)
            print(f"🗑️ Coleção existente {collection_name} deletada")
        except:
            pass
        
        # Criar nova coleção
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Dividir em lotes
        batch_size = BATCH_SIZE
        batches = []
        for i in range(0, len(unique_values), batch_size):
            batch_texts = unique_values[i:i + batch_size]
            batches.append((i // batch_size, batch_texts))
        
        print(f"📊 Processando {len(unique_values)} valores em {len(batches)} lotes de até {batch_size} itens")
        
        total_start_time = time.time()
        all_embeddings = []
        all_texts = []
        
        # Processar lotes em paralelo
        with ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(self.process_batch_parallel, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                try:
                    batch_id, texts, embeddings = future.result()
                    if embeddings and len(embeddings) == len(texts):
                        all_texts.extend(texts)
                        all_embeddings.extend(embeddings)
                    else:
                        print(f"⚠️ Lote {batch_id} falhou: {len(embeddings)} embeddings para {len(texts)} textos")
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"❌ Erro processando lote {batch[0]}: {e}")
        
        # Inserir na coleção
        if all_embeddings and all_texts:
            print(f"💾 Inserindo {len(all_embeddings)} embeddings na coleção {collection_name}...")
            
            ids = [f"{collection_name}_{i}" for i in range(len(all_texts))]
            
            collection.add(
                embeddings=all_embeddings,
                documents=all_texts,
                ids=ids
            )
            
            total_elapsed = time.time() - total_start_time
            avg_speed = len(all_embeddings) / total_elapsed if total_elapsed > 0 else 0
            
            print(f"✅ Coleção {collection_name} criada com {len(all_embeddings)} embeddings")
            print(f"⏱️ Tempo total: {total_elapsed:.2f}s ({avg_speed:.1f} embeddings/s)")
        else:
            print(f"❌ Falha ao criar embeddings para {collection_name}")
    
    def initialize_all_collections(self):
        """Inicializa todas as coleções categóricas"""
        collections_config = {
            CHROMA_COLLECTIONS["tipo"]: "tipo",
            CHROMA_COLLECTIONS["subtipo"]: "subtipo",
            CHROMA_COLLECTIONS["nome_unidade_organizacional"]: "nome_unidade_organizacional",
            CHROMA_COLLECTIONS["id_unidade_organizacional_mae"]: "id_unidade_organizacional_mae"
        }
        
        total_start_time = time.time()
        
        for collection_name, column_name in collections_config.items():
            try:
                self.initialize_collection(collection_name, column_name)
            except Exception as e:
                print(f"❌ Erro ao inicializar {collection_name}: {e}")
        
        total_elapsed = time.time() - total_start_time
        print(f"\n🏁 Inicialização completa em {total_elapsed:.2f}s")
        
        # Verificar status das coleções
        self.verify_collections()
    
    def verify_collections(self):
        """Verifica o status das coleções criadas"""
        print("\n🔍 Verificando coleções criadas:")
        
        total_embeddings = 0
        for collection_name in CHROMA_COLLECTIONS.values():
            try:
                collection = self.chroma_client.get_collection(collection_name)
                count = collection.count()
                total_embeddings += count
                print(f"  ✅ {collection_name}: {count} embeddings")
            except Exception as e:
                print(f"  ❌ {collection_name}: Erro - {e}")
        
        print(f"\n📊 Total de embeddings criados: {total_embeddings}")

def main():
    """Função principal para inicializar as coleções"""
    print("🎯 Iniciando inicialização de embeddings categóricos...")
    
    try:
        initializer = CategoryEmbeddingsInitializer()
        initializer.initialize_all_collections()
        print("\n🎉 Inicialização concluída com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        raise

if __name__ == "__main__":
    main()