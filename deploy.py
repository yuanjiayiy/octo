"""
deploy.py

Provide a lightweight server/client implementation for deploying Octo models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs Octo model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import argparse
import os.path

import numpy as np

# ruff: noqa: E402
import json_numpy
import cv2
from octo.model.octo_model import OctoModel
import jax
json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_octo_prompt(instruction: str, octo_path: Union[str, Path]) -> str:
    if "v01" in octo_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OctoServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for Octo models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """

        # Load Octo model
        self.model = OctoModel.load_pretrained(cfg.model_path, step=cfg.step)
        
    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            task = self.model.create_tasks(texts=[payload["instruction"]])
            def reshape_batch(payload):
                images = dict()
                for k in payload:
                    if k.startswith("image"):
                        original_img = payload[k]
                        og_img = Image.fromarray(original_img, 'RGB')
                        #og_img.save(f"original_{k}.jpg")
                        img_resized = cv2.resize(original_img, (256, 256))
                        resized_img = Image.fromarray(img_resized, 'RGB')
                        resized_img.save(f"resized_{k}.jpg")
                        img = img_resized[np.newaxis,np.newaxis,...]
                        images[k] = img
                    if k in ["proprio"]:
                        images[k] = payload[k]
                return images
            observation = {
                **reshape_batch(payload=payload),
                "timestep_pad_mask": np.array([[True]])
            }
            #print(observation)
            #import pdb; pdb.set_trace()
            action = self.model.sample_actions(observation,
                                               task,
                                               # unnormalization_statistics=self.model.dataset_statistics["action"],
                                               rng=jax.random.PRNGKey(0))
            #import pdb; pdb.set_trace()
            print(np.argmax(np.asarray(action)[0], axis=1))
            # import pdb; pdb.set_trace()
            action = np.argmax(np.asarray(action)[0][0])
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port
    model_path: str = "hf://rail-berkeley/octo-small-1.5"
    step: int = None


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OctoServer(cfg=cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
