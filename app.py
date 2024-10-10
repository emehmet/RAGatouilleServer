from fastapi import FastAPI
from pydantic import BaseModel
from ragatouille import RAGPretrainedModel

# RAG modelini yükle
RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2")

# Döküman indekslemek için kullanılacak FastAPI uygulaması
index_app = FastAPI()

class IndexRequest(BaseModel):
    full_document: str
    document_id: str
    metadata: dict
    index_name: str
    max_document_length: int = 4096  # Varsayılan değeri burada belirleyebilirsiniz
    split_documents: bool

@index_app.post("/index")
async def index_document(index_request: IndexRequest):
    RAG.index(
        collection=[index_request.full_document],
        document_ids=[index_request.document_id],
        document_metadatas=[index_request.metadata],
        index_name=index_request.index_name,
        max_document_length=index_request.max_document_length,
        split_documents=index_request.split_documents,
        # use_faiss=True
    )
    return {"status": "indexed", "document_id": index_request.document_id}

# Sorgu yapmak için kullanılacak FastAPI uygulaması
search_app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@search_app.post("/search")
async def search_rag(query_request: QueryRequest):
    # Sorguları ayırmak için özel bir karakter kullanıyoruz (örneğin '|')
    queries = query_request.query.split('|')
    
    # Çoklu sorguları RAG.search fonksiyonuna gönderiyoruz
    result = RAG.search(query=queries)
    
    # Sonuçları döndürüyoruz
    return {"result": result}
