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

load_dotenv()

class OpenAIClient:
    """Custom OpenAI client using requests"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_embeddings(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
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
        self.bq_client = bigquery.Client.from_service_account_json(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        self.openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
    def get_categorical_values(self, column_name: str, limit: int = 10000) -> List[str]:
        """Busca valores únicos de uma coluna categórica"""
        query = f"""
        SELECT DISTINCT {column_name}
        FROM `datario.adm_central_atendimento_1746.chamado`
        WHERE {column_name} IS NOT NULL 
        AND data_particao >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
        ORDER BY {column_name}
        LIMIT {limit}
        """
        
        print(f"🔍 Buscando valores únicos para {column_name}...")
        result = self.bq_client.query(query).to_dataframe()
        values = result[column_name].dropna().unique().tolist()
        print(f"✅ Encontrados {len(values)} valores únicos para {column_name}")
        return values
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Cria embeddings usando OpenAI com processamento otimizado em lotes"""
        print(f"🤖 Criando embeddings para {len(texts)} textos...")
        
        # Usar lotes maiores para melhor eficiência
        batch_size = 2000  # OpenAI permite até 2048 textos por request
        all_embeddings = []
        
        import time
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i, batch_start in enumerate(range(0, len(texts), batch_size)):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            
            print(f"  📦 Processando lote {i+1}/{total_batches} ({len(batch)} textos)...")
            
            try:
                embeddings = self.openai_client.create_embeddings(
                    texts=batch,
                    model="text-embedding-3-large"
                )
                all_embeddings.extend(embeddings)
                
                # Progress feedback
                processed = batch_end
                print(f"  ✅ {processed}/{len(texts)} textos processados ({processed/len(texts)*100:.1f}%)")
                
                # Rate limiting respeitoso (evitar 429 errors)
                if i < total_batches - 1:  # Não esperar no último lote
                    time.sleep(0.1)  # Pequena pausa entre requests
                    
            except Exception as e:
                print(f"  ❌ Erro no lote {i+1}: {e}")
                # Em caso de erro, tentar lote menor
                if len(batch) > 100:
                    print(f"  🔄 Tentando com lotes menores...")
                    small_batch_size = 100
                    for j in range(0, len(batch), small_batch_size):
                        small_batch = batch[j:j + small_batch_size]
                        try:
                            embeddings = self.openai_client.create_embeddings(
                                texts=small_batch,
                                model="text-embedding-3-large"
                            )
                            all_embeddings.extend(embeddings)
                            time.sleep(0.2)  # Pausa maior para recovery
                        except Exception as e2:
                            print(f"    ❌ Erro mesmo com lote pequeno: {e2}")
                            # Adicionar embeddings vazios para manter consistência
                            all_embeddings.extend([[0.0] * 3072] * len(small_batch))
                else:
                    print(f"    ❌ Lote já pequeno, adicionando embeddings vazios")
                    all_embeddings.extend([[0.0] * 3072] * len(batch))
        
    def create_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        """Versão alternativa com processamento paralelo usando threads"""
        import concurrent.futures
        import threading
        import time
        
        print(f"🚀 Criando embeddings em paralelo para {len(texts)} textos...")
        
        # Configuração para processamento paralelo
        batch_size = 1000  # Lotes grandes
        max_workers = 3    # Número de threads simultâneas
        all_embeddings = []
        lock = threading.Lock()
        
        def process_batch(batch_data):
            batch_idx, batch = batch_data
            try:
                embeddings = self.openai_client.create_embeddings(
                    texts=batch,
                    model="text-embedding-3-large"
                )
                
                with lock:
                    print(f"  ✅ Lote {batch_idx + 1} concluído ({len(batch)} textos)")
                
                return batch_idx, embeddings
                
            except Exception as e:
                print(f"  ❌ Erro no lote {batch_idx + 1}: {e}")
                return batch_idx, [[0.0] * 3072] * len(batch)
        
        # Criar lotes
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i // batch_size, batch))
        
        print(f"📦 Processando {len(batches)} lotes com até {max_workers} threads...")
        
        # Processar em paralelo
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch_data): batch_data[0] 
                             for batch_data in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx, embeddings = future.result()
                results[batch_idx] = embeddings
                
                # Rate limiting entre requests
                time.sleep(0.1)
        
        # Reordenar resultados
        for i in sorted(results.keys()):
            all_embeddings.extend(results[i])
        
        print(f"🎉 Processamento paralelo concluído: {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def initialize_collection(self, collection_name: str, values: List[str]):
        """Inicializa uma coleção ChromaDB"""
        print(f"📚 Inicializando coleção {collection_name}...")
        
        # Deletar coleção existente se houver
        try:
            self.chroma_client.delete_collection(collection_name)
            print(f"  Coleção {collection_name} existente removida")
        except Exception:
            pass
        
        # Criar nova coleção
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Criar embeddings (usar versão paralela para melhor performance)
        try:
            embeddings = self.create_embeddings_parallel(values)
        except Exception as e:
            print(f"⚠️ Erro com processamento paralelo: {e}")
            print("🔄 Tentando com processamento sequencial...")
            embeddings = self.create_embeddings(values)
        
        # Adicionar à coleção
        ids = [f"{collection_name}_{i}" for i in range(len(values))]
        collection.add(
            embeddings=embeddings,
            documents=values,
            ids=ids
        )
        
        print(f"✅ Coleção {collection_name} criada com {len(values)} documentos")
        return collection
    
    def initialize_all_collections(self):
        """Inicializa todas as coleções necessárias"""
        import time
        
        print("🚀 Inicializando todas as coleções de embeddings...\n")
        start_time = time.time()
        
        collections_config = {
            "nome_unidade_organizacional": "nome_unidade_organizacional",
            "id_unidade_organizacional_mae": "id_unidade_organizacional_mae", 
            "tipo": "tipo",
            "subtipo": "subtipo"
        }
        
        total_embeddings = 0
        successful_collections = 0
        
        for i, (collection_name, column_name) in enumerate(collections_config.items(), 1):
            try:
                print(f"📋 [{i}/{len(collections_config)}] Processando: {column_name}")
                print("-" * 50)
                
                col_start_time = time.time()
                values = self.get_categorical_values(column_name)
                
                if values:
                    print(f"📊 Valores únicos encontrados: {len(values)}")
                    self.initialize_collection(collection_name, values)
                    total_embeddings += len(values)
                    successful_collections += 1
                    
                    col_elapsed = time.time() - col_start_time
                    print(f"⏱️ Tempo da coleção: {col_elapsed:.2f}s")
                    print(f"🚀 Taxa: {len(values)/col_elapsed:.1f} embeddings/segundo")
                else:
                    print(f"⚠️ Nenhum valor encontrado para {column_name}")
                    
            except Exception as e:
                print(f"❌ Erro ao processar {column_name}: {e}")
            
            print("=" * 60)
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Inicialização concluída!")
        print(f"✅ Coleções criadas: {successful_collections}/{len(collections_config)}")
        print(f"📊 Total de embeddings: {total_embeddings}")
        print(f"⏱️ Tempo total: {elapsed_time:.2f}s")
        if total_embeddings > 0:
            print(f"🚀 Taxa média: {total_embeddings/elapsed_time:.1f} embeddings/segundo")
    
    def test_similarity_search(self, collection_name: str, query: str, n_results: int = 3):
        """Testa busca por similaridade"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Criar embedding da query
            embeddings = self.openai_client.create_embeddings(
                texts=[query],
                model="text-embedding-3-large"
            )
            query_embedding = embeddings[0]
            
            # Buscar
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            print(f"🔍 Teste de busca em {collection_name} para '{query}':")
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                similarity = 1 - distance  # Converter distância cosseno para similaridade
                print(f"  {i+1}. {doc} (similaridade: {similarity:.3f})")
            
        except Exception as e:
            print(f"❌ Erro no teste: {e}")

def main():
    """Função principal"""
    print("🤖 Inicializador de Embeddings para Categorias")
    print("=" * 50)
    
    initializer = CategoryEmbeddingsInitializer()
    
    # Inicializar todas as coleções
    initializer.initialize_all_collections()
    
    # Fazer alguns testes
    print("\n🧪 Executando testes de similaridade...\n")
    
    test_cases = [
        ("nome_unidade_organizacional", "iluminação"),
        ("tipo", "buraco na rua"),
        ("subtipo", "lâmpada queimada"),
        ("id_unidade_organizacional_mae", "ouvidoria")
    ]
    
    for collection, query in test_cases:
        initializer.test_similarity_search(collection, query)
        print()

if __name__ == "__main__":
    main()