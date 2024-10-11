from fastapi import FastAPI
from pydantic import BaseModel
from ragatouille import RAGPretrainedModel

# RAG modelini yükle
RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2")

# FastAPI uygulaması
app = FastAPI()

# İndeksleme isteği modeli
class IndexRequest(BaseModel):
    full_document: str
    document_id: str
    metadata: dict
    index_name: str
    max_document_length: int = 4096
    split_documents: bool

# İndeksleme endpoint'i
@app.post("/index")
async def index_document(index_request: IndexRequest):
    RAG.index(
        collection=[index_request.full_document],
        document_ids=[index_request.document_id],
        document_metadatas=[index_request.metadata],
        index_name=index_request.index_name,
        max_document_length=index_request.max_document_length,
        split_documents=index_request.split_documents,
        use_faiss=False
    )
    return {"status": "indexed", "document_id": index_request.document_id}

# Sorgu isteği modeli
class QueryRequest(BaseModel):
    query: str

# Sorgu endpoint'i
@app.post("/search")
async def search_rag(query_request: QueryRequest):
    queries = query_request.query.split('|')
    result = RAG.search(query=queries)
    return {"result": result}
