# 🤖 Agente de Análise de Dados - Rio de Janeiro

Agente de IA que responde perguntas sobre dados da prefeitura usando LangGraph e BigQuery.

## 🚀 Como usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar credenciais

Copie o arquivo de exemplo:
```bash
cp .env.example .env
```

Configure no arquivo `.env`:
```bash
# Sua chave OpenAI
OPENAI_API_KEY=sk-proj-sua-chave-aqui

# Caminho para seu arquivo JSON do BigQuery
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
```

### 3. Executar

**Teste rápido:**
```bash
python agent.py
```

**Teste detalhado:**
```bash
python test_agent.py
```

**Uso no código:**
```python
from agent import DataAnalystAgent

agent = DataAnalystAgent()
resposta = agent.run("Quantos chamados foram abertos hoje?")
print(resposta)
```

## � Perguntas de exemplo

O agente responde perguntas como:
- "Quantos chamados foram abertos no dia 28/11/2024?"
- "Qual o subtipo mais comum de Iluminação Pública?"
- "Quais os 3 bairros com mais chamados sobre reparo de buraco em 2023?"
- "Olá, tudo bem?"

## 🏗️ Arquitetura

O agente usa LangGraph com 4 nós principais:
1. **Router**: Classifica se a pergunta precisa de dados
2. **Gerador SQL**: Cria consultas otimizadas para BigQuery  
3. **Executor**: Executa a consulta e trata erros
4. **Sintetizador**: Converte dados em resposta natural

---

# Desafio Técnico Original

Bem-vindo(a) ao desafio técnico para a vaga de Pessoa Cientista de Dados Pleno no nosso time de transformação digital, focado em criar soluções inovadoras para a cidade do Rio de Janeiro!

## 🚀 Configuração e Instalação

### Pré-requisitos

- Python 3.8+
- Conta no Google Cloud Platform com acesso ao BigQuery
- Chave de API da OpenAI ou Anthropic

### 1. Clonar o repositório

```bash
git clone [url-do-repositorio]
cd desafio-pleno-data-scientist-ai
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Copie o arquivo de exemplo e configure suas chaves:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas credenciais:

```bash
# Chave da API OpenAI (recomendado)
OPENAI_API_KEY=sk-proj-...

# Ou chave da API Anthropic (alternativa)
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Configurar acesso ao BigQuery

**🔑 Você tem um arquivo JSON de conta de serviço?**

1. **Coloque o arquivo JSON na pasta do projeto**
2. **Configure no arquivo `.env`:**
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
   ```
3. **Verifique permissões**: Sua conta de serviço precisa de:
   - BigQuery Data Viewer  
   - BigQuery Job User

## 🎮 Como Usar

### Execução Básica

```bash
python agent.py
```

Este comando executará automaticamente as 6 perguntas de teste dos requisitos.

### Uso Programático

```python
from agent import DataAnalystAgent

# Inicializar agente
agent = DataAnalystAgent(llm_provider="openai")  # ou "anthropic"

# Fazer pergunta
resposta = agent.run("Quantos chamados foram abertos hoje?")
print(resposta)
```

### Exemplos de Perguntas Suportadas

**📊 Análises de Dados:**
- "Quantos chamados foram abertos no dia 28/11/2024?"
- "Qual o subtipo mais comum de Iluminação Pública?"
- "Quais os 3 bairros com mais chamados sobre reparo de buraco em 2023?"
- "Qual unidade organizacional mais atendeu chamados de fiscalização?"

**💬 Conversacionais:**
- "Olá, tudo bem?"
- "Obrigado pela ajuda!"
- "Me dê sugestões de brincadeiras para fazer com meu cachorro!"

## 🔧 Personalização

### Trocar Modelo de LLM

```python
# Usar OpenAI GPT-4
agent = DataAnalystAgent(llm_provider="openai")

# Usar Anthropic Claude
agent = DataAnalystAgent(llm_provider="anthropic")
```

### Modificar Prompts

Os prompts estão definidos nos métodos dos nós e podem ser personalizados:

- `router_node`: Classificação de intenção
- `sql_generator_node`: Geração de SQL
- `response_synthesizer_node`: Síntese de respostas
- `conversational_responder_node`: Respostas conversacionais

### Adicionar Novos Nós

```python
# Adicionar novo nó
workflow.add_node("novo_no", self.novo_no_method)

# Configurar roteamento
workflow.add_edge("origem", "novo_no")
```

## 🧪 Testes

### Perguntas de Validação dos Requisitos

O agente foi testado com as seguintes perguntas obrigatórias:

1. ✅ Quantos chamados foram abertos no dia 28/11/2024?
2. ✅ Qual o subtipo de chamado mais comum relacionado a "Iluminação Pública"?
3. ✅ Quais os 3 bairros que mais tiveram chamados abertos sobre "reparo de buraco" em 2023?
4. ✅ Qual o nome da unidade organizacional que mais atendeu chamados de "Fiscalização de estacionamento irregular"?
5. ✅ Olá, tudo bem?
6. ✅ Me dê sugestões de brincadeiras para fazer com meu cachorro!

### Executar Testes

```bash
python agent.py
```

## 🏛️ Dados Utilizados

### Tabelas do BigQuery

- **Chamados do 1746**: `datario.adm_central_atendimento_1746.chamado`
  - Dados de chamados de serviços públicos
  - Particionada por `data_particao` (otimização de performance)
  
- **Bairros**: `datario.dados_mestres.bairro`
  - Catálogo de bairros para enriquecimento de dados

### Otimizações de Performance

- ✅ Uso obrigatório de filtros de data em consultas
- ✅ Seleção específica de colunas (evita SELECT *)
- ✅ Uso de LIMIT para controlar tamanho dos resultados
- ✅ JOINs otimizados apenas quando necessário

## 🔍 Detalhes Técnicos

### Dependências Principais

```
langgraph==0.2.45          # Orquestração de grafos
langchain==0.3.7           # Framework base
langchain-openai==0.2.9    # Integração OpenAI
google-cloud-bigquery==3.27.0  # Cliente BigQuery
pandas==2.2.3              # Manipulação de dados
```

### Estado do Agente

```python
class AgentState(TypedDict):
    question: str                    # Pergunta original
    intent: Optional[str]            # Intenção classificada
    sql_query: Optional[str]         # SQL gerado
    data_result: Optional[pd.DataFrame]  # Resultado da consulta
    final_response: str              # Resposta final
    error: Optional[str]             # Erros capturados
    messages: List[Dict[str, str]]   # Histórico da conversa
```

### Tratamento de Erros

- **Erro de Autenticação BigQuery**: Orientação para configuração
- **Erro de SQL**: Resposta amigável sugerindo reformulação
- **Erro de API LLM**: Fallback com mensagem explicativa
- **Dados Vazios**: Resposta informativa sobre ausência de resultados

## 🎯 Diferenciais Implementados

### 🧠 Inteligência Avançada
- **Roteamento Inteligente**: Classificação automática de intenção
- **SQL Contextual**: Geração de consultas baseada no contexto da pergunta
- **Síntese Inteligente**: Conversão de dados em insights compreensíveis

### ⚡ Performance
- **Consultas Otimizadas**: Uso de partições e filtros para performance
- **Cache de Memória**: MemorySaver para contexto de conversação
- **Tratamento Assíncrono**: Preparado para operações não-bloqueantes

### 🛡️ Robustez
- **Tratamento Abrangente de Erros**: Captura e resposta amigável a falhas
- **Validação de Dados**: Verificação de resultados antes da síntese
- **Fallbacks Inteligentes**: Respostas úteis mesmo em cenários de erro

### 🔧 Flexibilidade
- **Multi-LLM**: Suporte a OpenAI e Anthropic
- **Configuração Flexível**: Fácil personalização de prompts e comportamentos
- **Extensibilidade**: Arquitetura permite adição de novos nós facilmente

## 📝 Próximos Passos

### Melhorias Futuras
- [ ] Cache de consultas SQL frequentes
- [ ] Validação de SQL antes da execução
- [ ] Interface web com Streamlit/Gradio
- [ ] Logs estruturados para monitoramento
- [ ] Testes automatizados com pytest
- [ ] Métricas de performance e usage

### Objetivo

O objetivo deste desafio é avaliar suas habilidades no desenho e desenvolvimento de soluções baseadas em IA Generativa. Você irá projetar e construir um agente autônomo capaz de interagir com uma base de dados da prefeitura, transformando perguntas em linguagem natural em insights acionáveis.

Avaliaremos sua capacidade de:
- Projetar uma arquitetura de agente de IA.
- Orquestrar tarefas complexas utilizando o framework **LangGraph**.
- Gerar e executar consultas SQL de forma segura e eficiente.
- Integrar LLMs (Large Language Models) com fontes de dados estruturadas (BigQuery).
- Escrever código limpo, bem documentado e robusto.

#### Observação

É esperado que você possa não ter tido contato prévio com todas as tecnologias solicitadas (como LangGraph, por exemplo), e isso é intencional. Parte da avaliação consiste em verificar sua capacidade de aprender rapidamente e aplicar novos conceitos. Por essa razão, o desafio tem uma duração de 7 dias, permitindo que você tenha tempo para estudar e desenvolver sua solução.

### Conjunto de Dados

O agente deverá consultar dados públicos do projeto `datario` no BigQuery. As tabelas principais para este desafio são:

- **Chamados do 1746:** Dados de chamados de serviços públicos.
  - Caminho: `datario.adm_central_atendimento_1746.chamado` (*)
- **Bairros do Rio de Janeiro:** Catálogo de bairros para enriquecimento dos dados.
  - Caminho: `datario.dados_mestres.bairro`
 
> (*) A coluna `data_particao` é a coluna de particionamento da tabela, ela é feita em cima de um trunc (DATE) da coluna `data_inicio`

### Ferramentas e Recursos

- **Linguagem e Framework:** Python e LangGraph.
- **Banco de Dados:** Google BigQuery. Você precisará de uma conta no GCP para consultar os dados.
- **LLM:** Fique à vontade para escolher o modelo de sua preferência (OpenAI, Google, Anthropic, etc.).
- **Bibliotecas Python:** `langchain`, `langgraph`, `google-cloud-bigquery`, `pandas`.

**Recursos Úteis:**
- **Tutorial de Acesso ao BigQuery:** [Como acessar dados no datario.rio](https://docs.dados.rio/tutoriais/como-acessar-dados/)
- **Documentação do LangGraph:** [LangChain Python Documentation](https://python.langchain.com/docs/langgraph)

### Etapas do Desafio

1.  **Configuração:** Siga o tutorial para criar sua conta no GCP e configurar a autenticação para o BigQuery.
2.  **Fork:** Faça um fork deste repositório.
3.  **Desenvolvimento do Agente:** Crie um agente em Python utilizando LangGraph que atenda aos critérios definidos no arquivo `requisitos_do_agente.md`. O agente deve ser capaz de receber uma pergunta em linguagem natural e orquestrar os passos para respondê-la.
4.  **Estrutura do Projeto:** Organize seu código de forma clara. Sugerimos uma estrutura que inclua:
    - Um arquivo principal para a lógica do agente (ex: `agent.py`).
    - Um arquivo `requirements.txt` com as dependências.
    - Um `README.md` detalhado para o seu projeto.
5.  **Documentação:** Atualize o `README.md` do seu repositório explicando a arquitetura da sua solução, como configurá-la (chaves de API, etc.) e como executá-la.
6.  **Entrega:** Faça commits incrementais à medida que avança. Ao finalizar, envie o link do seu repositório (privado) no GitHub para `brunoalmeida@prefeitura.rio` e `fernanda.scovino@prefeitura.rio`.

## Avaliação

Sua solução será avaliada com base nos seguintes critérios e pesos:

- **Qualidade do Código e Arquitetura do Agente (peso 3):** Clareza, modularidade, eficiência e a lógica do grafo construído em LangGraph.
- **Robustez e Tratamento de Erros (peso 2):** Como o agente lida com perguntas ambíguas, consultas que falham ou resultados inesperados.
- **Qualidade da Resposta e Eficiência da Consulta (peso 2):** A precisão da resposta final em linguagem natural e a qualidade do SQL gerado (evitar consultas desnecessariamente custosas).
- **Documentação (peso 1):** A clareza das instruções para rodar seu projeto e a explicação da sua solução.

**Dica:** Vá além do básico! Soluções que demonstrarem um raciocínio mais sofisticado, como validar o SQL gerado antes da execução, lidar com perguntas ambíguas pedindo esclarecimentos, ou implementar alguma forma de memória, serão vistas com grande diferencial.

## Dúvidas

Se tiver alguma dúvida, entre em contato pelo email `brunoalmeida@prefeitura.rio` e/ou `fernanda.scovino@prefeitura.rio`.

Boa sorte! Estamos ansiosos para ver sua solução.

---

**Prefeitura da Cidade do Rio de Janeiro**
