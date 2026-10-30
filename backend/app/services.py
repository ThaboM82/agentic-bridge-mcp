import os
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pgvector.sqlalchemy import CosineDistance
from app.models import DocumentChunk
from app.schemas import DocumentCreate, SearchQuery, SearchResultItem

# Initialize async OpenAI client (picks up OPENAI_API_KEY from environment automatically)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


async def generate_embedding(text: str) -> list[float]:
    """Generates a 1536-dimensional vector embedding using OpenAI."""
    response = await client.embeddings.create(
        input=[text.replace("\n", " ")], model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


async def create_document(db: AsyncSession, payload: DocumentCreate) -> DocumentChunk:
    """Generates vector embedding for document content and saves to database."""
    vector = await generate_embedding(payload.content)

    db_document = DocumentChunk(
        title=payload.title, content=payload.content, embedding=vector
    )
    db.add(db_document)
    await db.commit()
    await db.refresh(db_document)
    return db_document


async def search_documents(
    db: AsyncSession, payload: SearchQuery
) -> list[SearchResultItem]:
    """Performs a semantic similarity search using pgvector cosine distance."""
    query_vector = await generate_embedding(payload.query)

    # Use CosineDistance to rank results from most similar to least similar
    stmt = (
        select(
            DocumentChunk,
            CosineDistance(DocumentChunk.embedding, query_vector).label(
                "distance"
            ),
        )
        .order_by(CosineDistance(DocumentChunk.embedding, query_vector))
        .limit(payload.limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    search_results = []
    for doc, distance in rows:
        # Convert cosine distance to a similarity score (1 - distance)
        similarity_score = float(1.0 - distance)
        search_results.append(
            SearchResultItem(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                similarity_score=similarity_score,
            )
        )

    return search_results