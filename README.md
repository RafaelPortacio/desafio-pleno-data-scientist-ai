# Agente de Análise de Dados - Rio de Janeiro

Agente LangGraph que consulta dados da Prefeitura do Rio via BigQuery, com busca vetorial ChromaDB para melhor precisão nas consultas SQL.

## Estrutura do Projeto

```
desafio-pleno-data-scientist-ai/
├── src/
│   ├── config/           # Configurações e prompts
│   ├── nodes/            # Nodes do LangGraph (router, sql_generator, etc)
│   ├── tools/            # Busca por similaridade ChromaDB  
│   └── utils/            # Inicialização de embeddings
├── static/schemas/       # Schemas das tabelas BigQuery
├── credentials/          # Chave de serviço Google Cloud
├── agent.py             # Ponto de entrada principal
└── pyproject.toml       # Dependências (UV)
```

## Diferenciais da Implementação

### RAG com ChromaDB
- **Busca vetorial nas colunas categóricas**: Embeddings de todas as categorias únicas (tipos de chamado e etc)
- **Similaridade semântica**: Mapeia termos do usuário para valores exatos do banco para poder gerar queries de filtro precisas.
- **Exemplo**: "Iluminação" → "ILUMINACAO_PUBLICA"
- **4 coleções ChromaDB**

## Como Rodar

### 1. Instalar UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# ou https://docs.astral.sh/uv/getting-started/installation/ # Windows
```

### 2. Configurar Credenciais
```bash
# BigQuery: colocar service account JSON em credentials/
# OpenAI: criar .env com OPENAI_API_KEY
cp .env.example .env
```

### 3. Instalar Dependências
```bash
uv sync                    # Instala dependências
```

### 4. Inicializar ChromaDB (primeira vez)
```bash
uv run initialize-embeddings  # Cria embeddings das categorias
```

### 5. Rodar o Agente
```bash
uv run python agent.py     # Roda o agente
```

## Dependências Principais

- **LangGraph 0.6.5**: Orquestração de agentes
- **ChromaDB 0.4.24**: Busca vetorial
- **BigQuery**: Fonte de dados
- **OpenAI**: Embeddings e LLM