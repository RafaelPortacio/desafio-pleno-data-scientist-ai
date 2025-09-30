"""
Agente LangGraph 0.6.5 para análise de dados da Prefeitura do Rio de Janeiro
Versão refatorada com estrutura modular
"""

import warnings
import chromadb
import os
from typing import Dict, Optional, TypedDict, List, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Importar nodes
from src.nodes.router import router_node
from src.nodes.sql_generator import sql_generator_node
from src.nodes.sql_executor import sql_executor_node
from src.nodes.response_synthesizer import response_synthesizer_node
from src.nodes.conversational_responder import conversational_responder_node

# Importar tools
from src.tools.category_tools import CATEGORY_TOOLS

# Importar configurações
from src.config.settings import SUPPRESS_WARNINGS, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTIONS

# Importar utils para inicialização
from src.utils.initialize_embeddings import CategoryEmbeddingsInitializer

# Suprimir warnings se configurado
if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", module="chromadb")
    warnings.filterwarnings("ignore", message=".*telemetry.*")

class AgentState(TypedDict):
    """Estado do agente"""
    question: str
    route: str
    sql_query: Optional[str]
    data_result: Optional[List[Dict]]
    final_response: str
    error: Optional[str]
    messages: List[Dict[str, str]]
    tool_context: Optional[str]

class RioDataAgent:
    """Agente para análise de dados da Prefeitura do Rio de Janeiro"""
    
    def __init__(self):
        """Inicializa o agente com o grafo LangGraph"""
        print("🚀 Inicializando agente...")
        
        # Verificar e inicializar ChromaDB se necessário
        self._ensure_chromadb_initialized()
        
        # Construir o grafo
        self.graph = self._build_graph()
        print("✅ Agente inicializado com sucesso!")
    
    def _ensure_chromadb_initialized(self):
        """Verifica se as coleções ChromaDB existem e as inicializa se necessário"""
        try:
            print("🔍 Verificando coleções ChromaDB...")
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            
            # Verificar se todas as coleções existem
            missing_collections = []
            total_embeddings = 0
            
            for collection_key, collection_name in CHROMA_COLLECTIONS.items():
                try:
                    collection = chroma_client.get_collection(collection_name)
                    count = collection.count()
                    total_embeddings += count
                    print(f"  ✅ {collection_name}: {count} embeddings")
                except Exception:
                    missing_collections.append((collection_key, collection_name))
            
            # Se há coleções faltando, verificar pré-requisitos para inicialização
            if missing_collections:
                print(f"⚠️ Encontradas {len(missing_collections)} coleções faltando")
                
                # Verificar pré-requisitos
                if not self._check_initialization_prerequisites():
                    print("❌ Pré-requisitos não atendidos. Continuando sem busca por similaridade...")
                    print("\n� Para habilitar busca por similaridade:")
                    print("   1. Configure GOOGLE_APPLICATION_CREDENTIALS")
                    print("   2. Configure OPENAI_API_KEY") 
                    print("   3. Execute: python src/utils/initialize_embeddings.py")
                    return
                
                print("�🔄 Inicializando coleções ChromaDB automaticamente...")
                
                try:
                    initializer = CategoryEmbeddingsInitializer()
                    initializer.initialize_all_collections()
                    print("✅ Coleções ChromaDB inicializadas com sucesso!")
                except Exception as e:
                    print(f"❌ Erro na inicialização automática: {e}")
                    print("💡 Execute manualmente: python src/utils/initialize_embeddings.py")
            else:
                print(f"✅ Todas as coleções ChromaDB estão disponíveis ({total_embeddings} embeddings total)")
                
        except Exception as e:
            print(f"⚠️ Aviso: Erro ao verificar ChromaDB: {e}")
            print("💡 O agente continuará, mas a busca por similaridade pode não funcionar")
            print("💡 Para resolver, execute: python src/utils/initialize_embeddings.py")
    
    def _check_initialization_prerequisites(self):
        """Verifica se os pré-requisitos para inicialização estão atendidos"""
        print("🔍 Verificando pré-requisitos...")
        
        # Verificar OPENAI_API_KEY
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("  ❌ OPENAI_API_KEY não configurada")
            return False
        else:
            print("  ✅ OPENAI_API_KEY configurada")
        
        # Verificar GOOGLE_APPLICATION_CREDENTIALS
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not google_creds:
            print("  ❌ GOOGLE_APPLICATION_CREDENTIALS não configurada")
            return False
        elif not os.path.exists(google_creds):
            print(f"  ❌ Arquivo de credenciais não encontrado: {google_creds}")
            return False
        else:
            print("  ✅ GOOGLE_APPLICATION_CREDENTIALS configurada")
        
        return True
    
    def _build_graph(self) -> StateGraph:
        """Constrói o grafo de estados do LangGraph"""
        
        # Definir funções de roteamento
        def route_question(state: AgentState) -> str:
            """Rota baseada na decisão do router"""
            route = state.get("route", "conversational")
            if route == "data_query":
                return "sql_generator"
            else:
                return "conversational_responder"
        
        def check_sql_success(state: AgentState) -> str:
            """Verifica se o SQL foi gerado com sucesso"""
            if state.get("sql_query") and not state.get("error"):
                return "sql_executor"
            else:
                return "response_synthesizer"
        
        def check_execution_success(state: AgentState) -> str:
            """Verifica se a execução foi bem-sucedida"""
            return "response_synthesizer"
        
        # Criar grafo
        workflow = StateGraph(AgentState)
        
        # Adicionar nodes
        workflow.add_node("router", router_node)
        workflow.add_node("sql_generator", sql_generator_node)
        workflow.add_node("sql_executor", sql_executor_node)
        workflow.add_node("response_synthesizer", response_synthesizer_node)
        workflow.add_node("conversational_responder", conversational_responder_node)
        
        # Definir edges
        workflow.set_entry_point("router")
        
        # Router decide o fluxo
        workflow.add_conditional_edges(
            "router",
            route_question,
            {
                "sql_generator": "sql_generator",
                "conversational_responder": "conversational_responder"
            }
        )
        
        # SQL Generator para SQL Executor
        workflow.add_conditional_edges(
            "sql_generator",
            check_sql_success,
            {
                "sql_executor": "sql_executor",
                "response_synthesizer": "response_synthesizer"
            }
        )
        
        # SQL Executor para Response Synthesizer
        workflow.add_conditional_edges(
            "sql_executor",
            check_execution_success,
            {
                "response_synthesizer": "response_synthesizer"
            }
        )
        
        # Terminar nos nodes finais
        workflow.add_edge("response_synthesizer", END)
        workflow.add_edge("conversational_responder", END)
        
        # Compilar com memory
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    def run(self, question: str, config: Optional[Dict] = None) -> str:
        """
        Executa o agente com uma pergunta
        
        Args:
            question: Pergunta do usuário
            config: Configuração opcional para o grafo
            
        Returns:
            Resposta final do agente
        """
        # Estado inicial
        initial_state: AgentState = {
            "question": question,
            "route": "",
            "sql_query": None,
            "data_result": None,
            "final_response": "",
            "error": None,
            "messages": [{"role": "user", "content": question}],
            "tool_context": None
        }
        
        # Configuração padrão
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        try:
            # Executar grafo
            result = self.graph.invoke(initial_state, config=config)
            return result.get("final_response", "Erro: Não foi possível gerar resposta")
            
        except Exception as e:
            error_msg = f"Erro no agente: {str(e)}"
            print(f"❌ {error_msg}")
            return "Desculpe, ocorreu um erro interno. Tente novamente ou reformule sua pergunta."

def main():
    """Função principal para testar o agente"""
    print("🤖 Inicializando Agente de Dados da Prefeitura do Rio de Janeiro...")
    
    try:
        agent = RioDataAgent()
        
        # Loop interativo
        while True:
            question = input("\n👤 Sua pergunta (ou 'quit' para sair): ").strip()
            
            if question.lower() in ['quit', 'sair', 'exit']:
                print("👋 Obrigado por usar o agente!")
                break
            
            if not question:
                continue
            
            print("🤔 Processando...")
            response = agent.run(question)
            print(f"🤖 {response}")
            
    except Exception as e:
        print(f"❌ Erro ao inicializar agente: {e}")

if __name__ == "__main__":
    main()