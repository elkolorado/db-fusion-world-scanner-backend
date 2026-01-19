import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List

class TCGDetails(BaseModel):
    name: str        # The slug used in folders (e.g., 'dbs')
    tcg_id: int      # The numeric ID
    tcg_name: str    # The folder name for images (e.g., 'DragonBallSuper')
    nprobes: int    # Number of probes for FAISS

class EnvConfig(BaseModel):
    supported_tcgs: List[TCGDetails]
    card_images_path: Path
    details_service_url: str

class Settings:
    def __init__(self):
        self.app_env = os.getenv("APP_ENV", "dev")
        
        # 2. Load YAML file
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        
        # 3. Parse the specific environment data
        env_data = raw_config["environments"].get(self.app_env)
        if not env_data:
            raise ValueError(f"Environment '{self.app_env}' not found in config.yaml")
        
        self.config = EnvConfig(**env_data)

        # Allow overriding some values via environment variables (useful for containers)
        card_images_env = os.getenv("CARD_IMAGES_PATH")
        if card_images_env:
            try:
                self.config.card_images_path = Path(card_images_env)
            except Exception:
                self.config.card_images_path = Path(str(card_images_env))

        details_url_env = os.getenv("DETAILS_SERVICE_URL")
        if details_url_env:
            self.config.details_service_url = details_url_env

    @property
    def supported_tcgs(self) -> List[TCGDetails]:
        return self.config.supported_tcgs

    @property
    def images_path(self) -> Path:
        return self.config.card_images_path

    @property
    def details_url(self) -> str:
        return self.config.details_service_url

    def get_tcg_by_id(self, tcg_id: int) -> TCGDetails:
        for tcg in self.supported_tcgs:
            if tcg.tcg_id == tcg_id:
                return tcg
        return None

    def get_tcg_by_name(self, name: str) -> TCGDetails:
        name = name.lower()
        for tcg in self.supported_tcgs:
            if tcg.name == name:
                return tcg
        return None

settings = Settings()