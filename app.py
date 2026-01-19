from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx

from core.tcg_manager import tcg_manager

# Keep environment optimizations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1:5500/", "http://192.168.1.22:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # app-scoped resources
    app.state.matching_semaphore = asyncio.Semaphore(10)
    app.state.http_client = httpx.AsyncClient()

    # Load FAISS indexes into tcg_manager
    tcg_manager.load_indexes()


# include API router
from api.endpoints import router as api_router
app.include_router(api_router)



