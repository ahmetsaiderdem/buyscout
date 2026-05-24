import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="BuyScout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def parse_products(shopping_results: list) -> list:
    products = []
    for r in shopping_results:
        link = r.get("link") or r.get("product_link") or ""
        products.append({
            "title": r.get("title", ""),
            "price": r.get("price", ""),
            "thumbnail": r.get("thumbnail", ""),
            "link": link,
            "source": r.get("source", ""),
        })
    return products


@app.get("/")
async def root():
    return {"status": "ok", "service": "BuyScout API"}


@app.get("/search")
async def search(q: str):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY ortam degiskeni tanimlanmamis")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_shopping",
                "q": q,
                "api_key": SERPAPI_KEY,
                "hl": "tr",
                "gl": "tr",
                "num": 20,
            },
        )

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text[:200])
        except Exception:
            err = resp.text[:200]
        raise HTTPException(status_code=502, detail=f"SerpAPI hatasi ({resp.status_code}): {err}")

    data = resp.json()
    products = parse_products(data.get("shopping_results", []))
    return {"products": products, "query": q, "count": len(products)}


@app.post("/visual-search")
async def visual_search(
    image: UploadFile = File(...),
    text: str = Form(default=""),
):
    if not OPENAI_API_KEY or not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY ortam degiskeni tanimlanmamis")
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY ortam degiskeni tanimlanmamis")

    image_bytes = await image.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = image.content_type or "image/jpeg"

    prompt = (
        "Bu gorseldeki urunu tanimla. Sadece urun adini ve ana ozelliklerini yaz "
        "(marka, model, renk, tip). Kisa tut, en fazla 10 kelime. Turkce yaz."
    )
    if text:
        prompt += f" Kullanicinin notu: {text}"

    vision_resp = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=60,
    )

    identified = vision_resp.choices[0].message.content.strip()

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_shopping",
                "q": identified,
                "api_key": SERPAPI_KEY,
                "hl": "tr",
                "gl": "tr",
                "num": 20,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="SerpAPI hatasi")

    data = resp.json()
    products = parse_products(data.get("shopping_results", []))
    return {"products": products, "query": identified, "identified": identified, "count": len(products)}
