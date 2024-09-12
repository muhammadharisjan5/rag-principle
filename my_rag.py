

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import os

api_key = os.getenv('OPENAI_API_KEY')

class RAG:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = SimpleDirectoryReader(self.file_path).load_data()
        self.vector_store_index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.index.storage_context.persist(persist_dir="storage/cache/resume/sleep")
        storage_context = StorageContext.from_defaults(persist_dir="storage/cache/resume/sleep")
        self.load_index = load_index_from_storage(storage_context)
        self.query_engine = self.vector_store_index.as_query_engine()
        
    def get_response(self, questions):
        result = self.query_engine.query(questions)
        response = result.response
        return str(response)
    
    def get_pprint_response(self):
        response = self.get_response()
        return str(pprint_response(response, show_source=True))
        
    def get_retriever(self, similarity_top_k=3, question):
        retrever = VectorIndexRetriever(
                    index = self.vector_store_index,
                    similarity_top_k = similarity_top_k
                    )
        query_engine = RetrieverQueryEngine(retriever=retrever)
        response = query_engine.query(question)
        
    def get_similarity_post_processor(self, similarity_cutoff=0.75, similarity_top_k=3):
        s_processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        retrever = VectorIndexRetriever(
                        index = self.vector_store_index,
                        similarity_top_k = similarity_top_k
                    )
        query_engine = RetrieverQueryEngine(retriever= retrever, node_postprocessors=[s_processor])
        
