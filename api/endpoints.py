import os
import re
import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse

from services.vision import match_card as svc_match_card
from core import config as core_config
from core.tcg_manager import tcg_manager

router = APIRouter()

CARD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
ALLOWED_EXT = {"jpg", "png"}


@router.post("/matchCard")
async def match_card_api(request: Request, file: UploadFile = File(...), tcg_name: str = "Dragon Ball Fusion World"):
    image_bytes = await file.read()
    app = request.app

    sem = app.state.matching_semaphore
    http_client = app.state.http_client

    # Build mapping from configured TCGs for flexible matching
    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    mapping = {}
    for tcg in core_config.settings.supported_tcgs:
        # map both the slug (`name`) and the display/folder name (`tcg_name`) to the slug
        mapping[_normalize(tcg.name)] = tcg.name
        mapping[_normalize(tcg.tcg_name)] = tcg.name

    tcg_folder = mapping.get(_normalize(tcg_name), tcg_name.lower())

    faiss_index = tcg_manager.indexes.get(tcg_folder)
    filenames = tcg_manager.filenames.get(tcg_folder)

    if faiss_index is None or filenames is None:
        raise HTTPException(status_code=400, detail="Unsupported or unloaded TCG")

    async with sem:
        loop = asyncio.get_running_loop()
        best_match = await loop.run_in_executor(None, svc_match_card, image_bytes, faiss_index, filenames)

    if not best_match:
        return {"best_match": None, "error": "No match found"}

    card_market_id = os.path.splitext(best_match)[0]
    try:
        details_url = core_config.settings.details_url.rstrip("/")
        resp = await http_client.get(f"{details_url}/card/{card_market_id}")
        if resp.status_code == 200:
            return {"best_match": best_match, "card_details": resp.json()}
        else:
            return {"best_match": best_match, "error": "Failed to fetch card details"}
    except Exception as e:
        return {"best_match": best_match, "error": f"API Error: {str(e)}"}


@router.get("/card-image/{tcg_id}/{cardMarketId}.{extension}")
async def get_card_image(tcg_id: int, cardMarketId: str, extension: str):
    if extension not in ALLOWED_EXT:
        raise HTTPException(status_code=400)
    if not CARD_ID_RE.match(cardMarketId):
        raise HTTPException(status_code=400)

    tcg = core_config.settings.get_tcg_by_id(tcg_id)
    if not tcg:
        raise HTTPException(status_code=404)

    tcg_name = tcg.tcg_name
    base = os.path.abspath(str(core_config.settings.images_path))
    path = os.path.abspath(os.path.join(base, f"{tcg_name}/{cardMarketId}.{extension}"))
    if not path.startswith(base) or not os.path.exists(path):
        raise HTTPException(status_code=404)

    return FileResponse(
        path,
        media_type="image/jpeg" if extension == "jpg" else "image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/")
async def read_root():
    return {"Hello": "World"}
