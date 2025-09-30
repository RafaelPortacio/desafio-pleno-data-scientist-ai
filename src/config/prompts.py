"""
Configurações centralizadas do agente
"""

# Prompts Templates
ROUTER_PROMPT = """
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

SQL_GENERATOR_SYSTEM_PROMPT = """Você é um especialista em SQL para BigQuery. 

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

{agent_scratchpad}

Formato:
REASONING: [seu raciocínio]
SQL: [apenas o código SQL]"""

TOOL_CONTEXT_PROMPT = """Você é um especialista em SQL para BigQuery.

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
SQL: [apenas o código SQL válido]"""

RESPONSE_SYNTHESIZER_PROMPT = """
Você é um assistente especializado em análise de dados da Prefeitura do Rio de Janeiro.

Com base nos dados retornados da consulta SQL, forneça uma resposta clara e informativa em português.

DADOS DA CONSULTA:
{data_result}

PERGUNTA ORIGINAL:
{question}

INSTRUÇÕES:
- Seja claro e direto
- Use números formatados adequadamente
- Se não há dados, explique de forma educada
- Mantenha tom profissional mas acessível
- Foque na informação mais relevante

Resposta:
"""

CONVERSATIONAL_PROMPT = """
Você é um assistente de análise de dados da Prefeitura do Rio de Janeiro.

Pergunta: {question}

Responda de forma amigável e profissional. Se a pergunta não for relacionada a dados da prefeitura, seja educado mas redirecione para seu propósito principal.
"""

# Constantes
DEFAULT_SCHEMAS_PATH = {
    "chamado": "static/schemas/schema_chamado.txt",
    "bairro": "static/schemas/schema_bairro.txt"
}

FALLBACK_SQL_PATTERNS = {
    "iluminação": """
    SELECT subtipo, COUNT(*) as total
    FROM `datario.adm_central_atendimento_1746.chamado`
    WHERE (LOWER(tipo) LIKE '%iluminação%' OR LOWER(subtipo) LIKE '%lâmpada%' OR LOWER(subtipo) LIKE '%poste%')
    GROUP BY subtipo
    ORDER BY total DESC
    LIMIT 5
    """,
    "buraco": """
    SELECT b.nome as bairro, COUNT(*) as chamados
    FROM `datario.adm_central_atendimento_1746.chamado` c
    JOIN `datario.dados_mestres.bairro` b ON c.id_bairro = b.id_bairro
    WHERE (LOWER(c.tipo) LIKE '%pavimentação%' OR LOWER(c.subtipo) LIKE '%buraco%' OR LOWER(c.subtipo) LIKE '%via%')
    AND DATE(c.data_inicio) BETWEEN '2023-01-01' AND '2023-12-31'
    GROUP BY b.nome
    ORDER BY chamados DESC
    LIMIT 3
    """,
    "fiscalização_estacionamento": """
    SELECT nome_unidade_organizacional, COUNT(*) as total
    FROM `datario.adm_central_atendimento_1746.chamado`
    WHERE (LOWER(subtipo) LIKE '%fiscalização%' AND LOWER(subtipo) LIKE '%estacionamento%')
    GROUP BY nome_unidade_organizacional
    ORDER BY total DESC
    LIMIT 1
    """
}