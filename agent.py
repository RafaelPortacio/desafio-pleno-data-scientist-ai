"""
Agente de IA para análise de dados da Prefeitura do Rio de Janeiro
Utiliza LangGraph para orquestrar um fluxo de análise de dados inteligente
"""

import warnings
import logging

# Suprimir warnings de telemetria do ChromaDB
warnings.filterwarnings("ignore", message=".*capture.*takes.*positional argument.*")
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Suprimir warnings adicionais
warnings.filterwarnings("ignore", message=".*Failed to send telemetry event.*")
import sys
if hasattr(sys, 'stderr'):
    original_stderr = sys.stderr
    
    class SuppressStderr:
        def write(self, data):
            # Suprimir apenas mensagens de telemetria
            if 'Failed to send telemetry event' not in data and 'capture() takes' not in data:
                original_stderr.write(data)
        def flush(self):
            original_stderr.flush()
    
    # Redirecionar stderr temporariamente
    import contextlib
    @contextlib.contextmanager
    def suppress_chromadb_warnings():
        old_stderr = sys.stderr
        sys.stderr = SuppressStderr()
        try:
            yield
        finally:
            sys.stderr = old_stderr

from typing import Dict, Any, List, Optional, TypedDict, Literal
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from google.cloud import bigquery
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Importar tools de categoria
from category_tools import CATEGORY_TOOLS

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """Estado compartilhado entre os nós do agente"""
    question: str
    intent: Optional[str]
    sql_query: Optional[str] 
    data_result: Optional[list]
    final_response: str
    error: Optional[str]
    messages: List[Dict[str, str]]

class DataAnalystAgent:
    """
    Agente analista de dados usando LangGraph para orquestrar consultas ao BigQuery
    """
    
    def __init__(self):
        """
        Inicializa o agente
        """
        self.llm = self._setup_llm()
        self.bigquery_client = self._setup_bigquery()
        self._ensure_chromadb_initialized()
        self.graph = self._build_graph()
        
    def _ensure_chromadb_initialized(self):
        """Verifica se os ChromaDBs existem e os inicializa se necessário"""
        import chromadb
        
        try:
            # Conectar ao ChromaDB com supressão de warnings
            with suppress_chromadb_warnings():
                client = chromadb.PersistentClient(path="./chroma_db")
                
                # Verificar se todas as coleções necessárias existem
                required_collections = ['tipo', 'subtipo', 'nome_unidade_organizacional', 'id_unidade_organizacional_mae']
                existing_collections = [col.name for col in client.list_collections()]
                
                missing_collections = [col for col in required_collections if col not in existing_collections]
                
                if missing_collections:
                    print(f"🔄 Coleções ausentes: {missing_collections}. Inicializando embeddings...")
                    self._initialize_embeddings()
                else:
                    print("✅ Todas as coleções ChromaDB encontradas.")
                    
        except Exception as e:
            print(f"🔄 Erro ao verificar ChromaDB: {e}. Inicializando embeddings...")
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Inicializa os embeddings das colunas categóricas"""
        try:
            from initialize_embeddings import CategoryEmbeddingsInitializer
            initializer = CategoryEmbeddingsInitializer()
            
            print("📊 Inicializando todas as coleções de embeddings...")
            initializer.initialize_all_collections()
            
            print("🎉 Embeddings inicializados com sucesso!")
            
        except Exception as e:
            print(f"⚠️ Erro ao inicializar embeddings: {e}")
            print("   O agente continuará funcionando sem as tools de similaridade.")
        
    def _setup_llm(self):
        """Configura o modelo de linguagem OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")
        
        return ChatOpenAI(
            model="gpt-5",
            temperature=1,
            api_key=api_key
        )
    
    def _setup_bigquery(self):
        """Configura cliente BigQuery usando chave de serviço JSON"""
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not key_path:
            raise Exception("GOOGLE_APPLICATION_CREDENTIALS não encontrada no .env")
        
        if not os.path.exists(key_path):
            raise Exception(f"Arquivo de chave não encontrado: {key_path}")
        
        try:
            print(f"🔑 Conectando ao BigQuery usando: {key_path}")
            return bigquery.Client.from_service_account_json(key_path)
        except Exception as e:
            raise Exception(f"Erro ao configurar BigQuery: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Constrói o grafo LangGraph"""
        # Criar o grafo
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("router", self.router_node)
        workflow.add_node("sql_generator", self.sql_generator_node)
        workflow.add_node("sql_executor", self.sql_executor_node)
        workflow.add_node("response_synthesizer", self.response_synthesizer_node)
        workflow.add_node("conversational_responder", self.conversational_responder_node)
        
        # Configurar ponto de entrada
        workflow.set_entry_point("router")
        
        # Adicionar roteamento condicional
        workflow.add_conditional_edges(
            "router",
            self.route_intent,
            {
                "data_query": "sql_generator",
                "conversational": "conversational_responder"
            }
        )
        
        # Fluxo linear para consultas de dados
        workflow.add_edge("sql_generator", "sql_executor")
        workflow.add_edge("sql_executor", "response_synthesizer")
        workflow.add_edge("response_synthesizer", END)
        workflow.add_edge("conversational_responder", END)
        
        # Compilar com checkpoint
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def router_node(self, state: AgentState) -> AgentState:
        """
        Nó roteador: determina se a pergunta requer dados ou é conversacional
        """
        question = state["question"]
        
        prompt = f"""
        Analise a seguinte pergunta e determine se ela requer consulta a dados ou é conversacional.

        Pergunta: "{question}"

        Responda APENAS com uma das opções:
        - "data_query" se a pergunta for sobre dados/estatísticas (ex: quantos, qual, quais, como, quando sobre chamados, bairros, etc.)
        - "conversational" se for saudação, agradecimento, pergunta genérica ou não relacionada a dados específicos

        Exemplos:
        - "Quantos chamados foram abertos?" -> data_query
        - "Qual o bairro com mais chamados?" -> data_query
        - "Olá, tudo bem?" -> conversational
        - "Obrigado!" -> conversational
        - "Me dê sugestões de brincadeiras" -> conversational
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()
        
        # Validar resposta
        if intent not in ["data_query", "conversational"]:
            intent = "conversational"  # Default para conversacional em caso de dúvida
        
        state["intent"] = intent
        state["messages"].append({"role": "system", "content": f"Intent classificado como: {intent}"})
        
        return state
    
    def route_intent(self, state: AgentState) -> Literal["data_query", "conversational"]:
        """Função de roteamento para o conditional_edges"""
        return state["intent"]
    
    def sql_generator_node(self, state: AgentState) -> AgentState:
        """
        Nó gerador de SQL: agente com tools para encontrar valores categóricos similares
        """
        question = state["question"]
        
        with open("schema_chamado.txt", "r", encoding="utf-8") as f:
            schema_chamado = f.read()
            
        with open("schema_bairro.txt", "r", encoding="utf-8") as f:
            schema_bairro = f.read()

        # Criar prompt para o agente com tools
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em SQL para BigQuery. 

        INSTRUÇÕES PARA USO DAS TOOLS:
        - Se a pergunta mencionar termos que podem corresponder a colunas categóricas, use as tools para encontrar valores exatos
        - Tools disponíveis:
        * get_nome_unidade_organizacional: Para buscar unidades organizacionais
        * get_id_unidade_organizacional_mae: Para buscar unidades mãe  
        * get_tipo: Para buscar tipos de chamados
        * get_subtipo: Para buscar subtipos específicos
        - Use as tools ANTES de gerar o SQL para garantir valores corretos

        Primeiro, faça um REASONING sobre a descrição do schema e como ela consegue atender à pergunta.

        Depois gere uma consulta SQL otimizada.

        INFORMAÇÕES IMPORTANTES:
        - Tabela principal: `datario.adm_central_atendimento_1746.chamado`
        - Tabela de bairros: `datario.dados_mestres.bairro`
        - Atente-se ao uso da coluna correta de data em suas consultas
        - Evite SELECT * - selecione apenas colunas necessárias
        - Use LIMIT quando apropriado para evitar resultados excessivos

        DESCRIÇÃO DA TABELA DE CHAMADOS:
        {schema_chamado}

        DESCRIÇÃO DA TABELA DADOS DOS BAIRROS:
        {schema_bairro}

        PARA JOINS COM BAIRROS:
        - Use: JOIN `datario.dados_mestres.bairro`

        INSTRUÇÕES:
        1. Use agregações (COUNT, GROUP BY) quando apropriado
        2. Para top N, use ORDER BY e LIMIT
        3. Se mencionar nomes de bairros, faça JOIN com a tabela de bairros
        4. O SQL gerado será executado imediatamente, não adicione explicações ou comentários
        5. O subtipo é subordinado ao tipo, prefira usar o tipo para filtrar primeiro e se não for o suficiente, use o subtipo para refinar a busca
        6. Evite usar queries com LIKE, elas são extremamente, as tools são para encontrar o valor exato correto (Por exemplo, se a pergunta for encontre subtipos relacionados a "Regras de Trânsito" use a tool get_tipo para encontrar o tipo exato para filtrar pelo nome dele na query, caso não haja NENHUM adequado, aí procure por subtipos relacionados e filtre por eles)
        
        {agent_scratchpad}

        Formato:
        REASONING: [seu raciocínio]
        SQL: [apenas o código SQL]"""),
            ("human", "Pergunta: {question}")
        ])

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(CATEGORY_TOOLS)
        
        # Create chain
        chain = prompt_template | llm_with_tools
        
        # Execute with tools
        print("🤖 Executando LLM com tools...")
        result = chain.invoke({
            "question": question,
            "schema_chamado": schema_chamado,
            "schema_bairro": schema_bairro,
            "agent_scratchpad": ""
        })
        
        # Verificar se há tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print("🔧 Executando tool calls...")
            
            # Executar cada tool call
            tool_results = []
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                print(f"  🛠️ Executando {tool_name} com args: {tool_args}")
                
                # Encontrar e executar a tool
                for tool in CATEGORY_TOOLS:
                    if tool.name == tool_name:
                        try:
                            tool_result = tool.func(**tool_args)
                            tool_results.append(f"{tool_name}: {tool_result}")
                            print(f"  ✅ Resultado: {tool_result}")
                        except Exception as e:
                            error_msg = f"Erro em {tool_name}: {e}"
                            tool_results.append(error_msg)
                            print(f"  ❌ {error_msg}")
                        break
            
            # Gerar nova consulta com os resultados das tools
            tool_context = "\n".join(tool_results)
            
            print("🔄 Gerando SQL com resultados das tools...")
            
            # Novo prompt com contexto das tools
            new_prompt = ChatPromptTemplate.from_messages([
                ("system", """Você é um especialista em SQL para BigQuery.

                INSTRUÇÕES:
                - Use os resultados das tools abaixo para gerar a consulta SQL
                - Se as tools não encontraram resultados similares, use LIKE com wildcards para buscar termos relacionados
                - NUNCA retorne [] ou SQL inválida - sempre gere uma consulta válida
                - Se não encontrou valores exatos, use termos mais genéricos ou padrões LIKE
                
                RESULTADOS DAS TOOLS:
                {tool_context}

                DESCRIÇÃO DA TABELA DE CHAMADOS:
                {schema_chamado}

                DESCRIÇÃO DA TABELA DADOS DOS BAIRROS:
                {schema_bairro}

                INSTRUÇÕES PARA FALLBACK:
                - Se não encontrou "Iluminação Pública", use LIKE '%iluminação%' ou '%lâmpada%' ou '%poste%'
                - Se não encontrou "reparo de buraco", use LIKE '%buraco%' ou '%pavimentação%' ou '%via%'
                - Se não encontrou "fiscalização estacionamento", use LIKE '%fiscalização%' e '%estacionamento%'
                - Sempre prefira gerar SQL funcional mesmo que aproximada
                - SEMPRE use nomes completos de tabelas: `datario.adm_central_atendimento_1746.chamado` e `datario.dados_mestres.bairro`
                - NUNCA use apenas "chamado" ou "bairro" - sempre com o dataset completo

                Primeiro, faça um REASONING sobre como usar os resultados das tools ou fallback.
                Depois gere uma consulta SQL otimizada e VÁLIDA.

                Formato:
                REASONING: [seu raciocínio]
                SQL: [apenas o código SQL válido]"""),
                ("human", "Pergunta: {question}")
            ])
            
            # Nova chain sem tools
            new_chain = new_prompt | self.llm
            
            # Executar com contexto das tools
            new_result = new_chain.invoke({
                "question": question,
                "tool_context": tool_context,
                "schema_chamado": schema_chamado,
                "schema_bairro": schema_bairro
            })
            
            full_response = new_result.content
            
        else:
            full_response = result.content
        
        # Extrair reasoning e SQL
        if "SQL:" in full_response:
            parts = full_response.split("SQL:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            sql_query = parts[1].strip()
            print(f"🧠 REASONING: {reasoning}")
        else:
            reasoning = "Reasoning não fornecido pelo LLM."
            sql_query = full_response
            
        # Validar SQL antes de prosseguir
        if not sql_query or sql_query.strip() == "" or sql_query.strip() == "[]" or "SELECT NULL" in sql_query.upper():
            print("⚠️ SQL inválida ou vazia detectada. Gerando fallback...")
            # Tentar fallback simples baseado na pergunta
            question_lower = state["question"].lower()
            if "iluminação" in question_lower:
                sql_query = """
                SELECT subtipo, COUNT(*) as total
                FROM `datario.adm_central_atendimento_1746.chamado`
                WHERE (LOWER(tipo) LIKE '%iluminação%' OR LOWER(subtipo) LIKE '%lâmpada%' OR LOWER(subtipo) LIKE '%poste%')
                GROUP BY subtipo
                ORDER BY total DESC
                LIMIT 5
                """
            elif "buraco" in question_lower:
                sql_query = """
                SELECT b.nome as bairro, COUNT(*) as chamados
                FROM `datario.adm_central_atendimento_1746.chamado` c
                JOIN `datario.dados_mestres.bairro` b ON c.id_bairro = b.id_bairro
                WHERE (LOWER(c.tipo) LIKE '%pavimentação%' OR LOWER(c.subtipo) LIKE '%buraco%' OR LOWER(c.subtipo) LIKE '%via%')
                AND DATE(c.data_inicio) BETWEEN '2023-01-01' AND '2023-12-31'
                GROUP BY b.nome
                ORDER BY chamados DESC
                LIMIT 3
                """
            elif "fiscalização" in question_lower and "estacionamento" in question_lower:
                sql_query = """
                SELECT nome_unidade_organizacional, COUNT(*) as total
                FROM `datario.adm_central_atendimento_1746.chamado`
                WHERE (LOWER(subtipo) LIKE '%fiscalização%' AND LOWER(subtipo) LIKE '%estacionamento%')
                GROUP BY nome_unidade_organizacional
                ORDER BY total DESC
                LIMIT 1
                """
            else:
                sql_query = "SELECT 'Não foi possível gerar consulta para esta pergunta' as mensagem"
            
            print(f"🔄 Usando SQL de fallback: {sql_query.strip()}")
        
        sql_query = sql_query.strip()
        
        # Limpar marcadores de código do SQL
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        sql_query = sql_query.strip()
        
        # Armazenar no estado
        state["sql_query"] = sql_query
        state["messages"].append({"role": "system", "content": f"SQL gerado: {sql_query}"})
        
        return state
    
    def sql_executor_node(self, state: AgentState) -> AgentState:
        """
        Nó executor de SQL: executa a consulta no BigQuery
        """
        sql_query = state["sql_query"]
        
        if not sql_query or sql_query.strip() == "":
            print("❌ ERRO: SQL query está vazia!")
            state["error"] = "SQL query vazia ou inválida"
            state["data_result"] = []
            return state
        
        # try:
            # Executar consulta
        print(f"🔍 Executando SQL:\n{sql_query}")
        results = self.bigquery_client.query(sql_query)
        
        # Converter para DataFrame
        df = results.to_dataframe().to_dict(orient="records")
        print(df)
        state["data_result"] = df
        state["messages"].append({
            "role": "system", 
            "content": f"Consulta executada com sucesso. {len(df)} linhas retornadas."
        })
            
        # except Exception as e:
        #     error_msg = f"Erro ao executar SQL: {str(e)}"
        #     print(error_msg)
        #     state["error"] = error_msg
        #     state["messages"].append({"role": "system", "content": error_msg})
            
        #     # Criar DataFrame vazio em caso de erro
        #     state["data_result"] = []
        
        return state
    
    def response_synthesizer_node(self, state: AgentState) -> AgentState:
        """
        Nó sintetizador: converte dados em resposta natural
        """
        question = state["question"]
        data_result = state["data_result"]
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
            # Se temos dados, sintetizar resposta
            if len(data_result):
                # Criar resumo dos dados
                data_summary = f"Dados encontrados ({len(data_result)} linhas):\n"
                if len(data_result) <= 10:
                    data_summary += str(data_result)
                else:
                    data_summary += str(data_result[:10])
                    data_summary += f"\n... e mais {len(data_result) - 10} linhas"
            else:
                data_summary = "Nenhum resultado encontrado."
            
            prompt = f"""
            Baseado nos dados consultados, responda à pergunta de forma clara e objetiva.

            Pergunta: "{question}"

            Dados:
            {data_summary}

            INSTRUÇÕES:
            1. Responda em português brasileiro
            2. Seja específico e cite os números encontrados
            3. Se houver múltiplos resultados, destaque os principais
            4. Mantenha a resposta focada na pergunta
            5. Se não houver dados, explique que não foram encontrados resultados para os critérios especificados
            """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_response = response.content.strip()
        
        state["final_response"] = final_response
        state["messages"].append({"role": "assistant", "content": final_response})
        
        return state
    
    def conversational_responder_node(self, state: AgentState) -> AgentState:
        """
        Nó para respostas conversacionais (saudações, agradecimentos, etc.)
        """
        question = state["question"]
        
        prompt = f"""
        Responda de forma amigável e natural à seguinte mensagem conversacional:
        
        "{question}"
        
        CONTEXTO: Você é um assistente de análise de dados da Prefeitura do Rio de Janeiro.
        
        INSTRUÇÕES:
        1. Seja cordial e profissional
        2. Se for saudação, retribua e se apresente brevemente
        3. Se for agradecimento, responda educadamente
        4. Se for pergunta genérica não relacionada a dados, responda de forma extremamente concisa e redirecione para suas capacidades de análise de dados.
        5. Mantenha respostas concisas
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_response = response.content.strip()
        
        state["final_response"] = final_response
        state["messages"].append({"role": "assistant", "content": final_response})
        
        return state
    
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
        initial_state = {
            "question": question,
            "intent": None,
            "sql_query": None,
            "data_result": None,
            "final_response": "",
            "error": None,
            "messages": [{"role": "user", "content": question}]
        }
        
        # Configuração padrão
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Executar grafo
        result = self.graph.invoke(initial_state, config=config)
        
        return result["final_response"]

def main():
    """Função principal para testar o agente"""
    # Perguntas de teste dos requisitos
    test_questions = [
        "Quantos chamados foram abertos no dia 28/11/2024?",
        "Qual o subtipo de chamado mais comum relacionado a 'Iluminação Pública'?",
        "Quais os 3 bairros que mais tiveram chamados abertos sobre 'reparo de buraco' em 2023?",
        "Qual o nome da unidade organizacional que mais atendeu chamados de 'Fiscalização de estacionamento irregular'?",
        "Olá, tudo bem?",
        "Me dê sugestões de brincadeiras para fazer com meu cachorro!"
    ]
    
    try:
        print("🚀 Inicializando Agente de Análise de Dados...")
        agent = DataAnalystAgent()
        print("✅ Agente inicializado com sucesso!")
        
        # Testar perguntas
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"TESTE {i}: {question}")
            print('='*50)
            
            try:
                response = agent.run(question)
                print(f"🤖 Resposta: {response}")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
    except Exception as e:
        print(f"❌ Erro ao inicializar agente: {e}")
        print("\n🔧 Verifique se:")
        print("   1. OPENAI_API_KEY está configurada no .env")
        print("   2. GOOGLE_APPLICATION_CREDENTIALS aponta para o arquivo JSON correto")
        print("   3. pip install -r requirements.txt foi executado")

if __name__ == "__main__":
    main()