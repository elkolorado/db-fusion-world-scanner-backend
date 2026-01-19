# DB Fusion World Scanner — README

**Overview**
- Small FastAPI service that matches card images against FAISS indexes using SIFT descriptors.
- Routes live under an API router at `api/endpoints.py`. Heavy image work is implemented in `services/vision.py` and index generation in `services/generate_indexes.py`.

**Development**

- Install deps:

```bash
pip install -r requirements.txt
```

- Configure the app:
  - Edit `config.yaml` to add the `card_images` folder and other settings, or set the env var `CARD_IMAGES_PATH` to point to your card images directory.
  - Optionally set `DETAILS_SERVICE_URL` to point to the external card-details service.

- Run (development):

```bash
python initialize.py
```

**Endpoints**

- POST `/matchCard`
  - Multipart `file` (image), query `tcg_name` (human-friendly name). Returns `best_match` (filename) and attempts to fetch `card_details` from the configured details service.

- GET `/card-image/{tcg_id}/{cardMarketId}.{extension}`
  - Serves card images from the configured `card_images` location.

- GET `/`
  - Health / hello endpoint.

**Services & Core components**

- `services/vision.py`
  - `extract_keypoints_from_bytes(image_bytes)` — loads bytes, fixes orientation, resizes, computes SIFT keypoints/descriptors.
  - `match_card(image_bytes, faiss_index, filenames)` — queries FAISS and performs Lowe ratio + voting to pick best filename.

- `services/generate_indexes.py`
  - `generate_for_tcg(slug)` and `generate_all()` — build FAISS indexes from card images and save `indexes/{slug}/faiss_index.bin` and `filenames.pkl`.

- `core/tcg_manager.py`
  - `tcg_manager` singleton that loads indexes at startup (used by endpoints).

- `core/config.py`
  - `settings` loads `config.yaml` and allows `CARD_IMAGES_PATH` / `DETAILS_SERVICE_URL` env overrides (useful for mounts/containers).

**Testing**

Run tests with unittest or pytest:

```bash
python -m unittest discover -s tests -p "test_*.py"
# or
pytest -q
```


