from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from src.config import OPENAI_EMBEDDING_MODEL, pg_url

COLLECTION = "baseline"


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    return PGVector(
        embeddings=OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL),
        connection=pg_url(),
        collection_name=COLLECTION,
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )
