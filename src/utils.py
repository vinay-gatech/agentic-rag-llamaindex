from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


load_dotenv()
documents   = SimpleDirectoryReader(input_files=['metagpt.pdf']).load_data()

splitter    = SentenceSplitter(chunk_size=1024)
nodes       = splitter.get_nodes_from_documents(documents)

Settings.llm            = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model    = OpenAIEmbedding(model="text-embedding-ada-002")

summary_index           = SummaryIndex(nodes)
vector_index            = VectorStoreIndex(nodes)

summary_query_engine    = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine     = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

query_engine    = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    # list of tools to invoke
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True,
)