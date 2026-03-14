import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Sample knowledge base articles
KB_ARTICLES = [
    {
        "title": "How to reset your password",
        "content": "To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter your email 4. Check inbox for reset link 5. Click link and set new password. Link expires in 24 hours."
    },
    {
        "title": "How to update payment method",
        "content": "To update payment: 1. Go to Settings 2. Click Billing 3. Click 'Update Payment Method' 4. Enter new card details 5. Click Save. Changes take effect immediately."
    },
    {
        "title": "How to cancel subscription",
        "content": "To cancel: 1. Go to Settings 2. Click Subscription 3. Click Cancel Plan 4. Confirm cancellation. You keep access until end of billing period. No refunds for partial months."
    },
    {
        "title": "How to download invoice",
        "content": "To download invoice: 1. Go to Settings 2. Click Billing 3. Click Invoice History 4. Click Download next to any invoice. Invoices are in PDF format."
    },
    {
        "title": "API rate limits",
        "content": "API limits by plan: Free: 100 requests/day. Pro: 10,000 requests/day. Enterprise: unlimited. If you hit limits you get 429 error. Upgrade plan to increase limits."
    },
    {
        "title": "How to invite team members",
        "content": "To invite team: 1. Go to Settings 2. Click Team 3. Click Invite Member 4. Enter email 5. Select role (Admin/Member) 6. Click Send Invite. They receive email to join."
    },
    {
        "title": "Two factor authentication setup",
        "content": "To enable 2FA: 1. Go to Settings 2. Click Security 3. Click Enable 2FA 4. Scan QR code with authenticator app 5. Enter 6-digit code to confirm. Use Google Authenticator or Authy."
    },
    {
        "title": "How to export data",
        "content": "To export your data: 1. Go to Settings 2. Click Data Export 3. Select date range 4. Choose format (CSV or JSON) 5. Click Export. File emailed to you within 1 hour."
    },
]

COLLECTION_NAME = "support_knowledge_base"
VECTOR_SIZE = 1536  # text-embedding-3-small dimension


async def seed():
    print("🚀 Starting knowledge base seeding...")

    # Connect to Qdrant
    client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333")
    )

    # Create collection
    print("📦 Creating collection...")
    await client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Embed articles
    print("🔢 Embedding articles...")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    points = []
    for i, article in enumerate(KB_ARTICLES):
        # Combine title + content for richer embedding
        text = f"{article['title']}\n{article['content']}"
        vector = await embedder.aembed_query(text)

        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={
                "title": article["title"],
                "content": article["content"],
            }
        ))
        print(f"  ✅ Embedded: {article['title']}")

    # Upload to Qdrant
    print("⬆️  Uploading to Qdrant...")
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"\n🎉 Done! {len(points)} articles loaded into Qdrant.")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    asyncio.run(seed())