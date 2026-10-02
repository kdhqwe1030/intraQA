from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from src.lib.config import OPENAI_EMBEDDING_MODEL, pg_url


def get_vector_store(collection: str, pre_delete_collection: bool = False) -> PGVector:
    """collection은 인덱스(청킹 방식) 이름과 같다. 예: baseline, structured"""
    return PGVector(
        embeddings=OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL),
        connection=pg_url(),
        collection_name=collection,
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )
