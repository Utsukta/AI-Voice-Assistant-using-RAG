import os
import tempfile
import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import LLMRerank

import nest_asyncio
nest_asyncio.apply()


# Function to handle file upload
def upload_file():
    loaded_file = SimpleDirectoryReader(input_dir="./Data").load_data()
    return loaded_file


# Function to initialize the vector store
def initialize_vector_store():
    client = qdrant_client.QdrantClient(location=':memory:')
    vector_store = QdrantVectorStore(client=client, collection_name="sampledata")
    return vector_store

# Main function
def RAG():
    # Initialize models and settings
    Settings.llm = Ollama(model="llama3", request_timeout=400.0)
    Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed")
    Settings.text_splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model)

    file = upload_file()
    

    vector_store = initialize_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=file,
        storage_context=storage_context,
        show_progress=True,
        transformations=[Settings.text_splitter]
    )

    index.storage_context.persist(persist_dir="dir")
    # reranker = LLMRerank( choice_batch_size=6,  top_n=2, )

    query_engine = index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
        similarity_top_k=10,
        # node_postprocessors=[
        # #     LLMRerank(
        # #     choice_batch_size=5,
        # #     top_n=2,
        # # )
        #     reranker
        # ]
    )
    return query_engine

