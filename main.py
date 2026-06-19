from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from typing import Any, Dict, Literal, Optional, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google import genai
from PIL import Image
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vastra_ai")

app = FastAPI(title="VastraAI Core Engine")
templates = Jinja2Templates(directory="templates")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_URL = "https://api.runware.ai/v1"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Reusable HTTP client for faster network performance (avoids repeated SSL handshakes)
http_client = httpx.AsyncClient(timeout=120)

ANALYSIS_MODEL = "gemini-3.1-flash-lite"
DisplayType = Literal["model", "mannequin", "flat", "hanging"]

# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    reference_image: str  
    display_types: List[str] = ["model"]  
    catalog: Optional[Dict[str, Any]] = None  
    prompt_override: Optional[str] = None  

class RunwareError(RuntimeError):
    pass

class CostTracker:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_cost = 0.0
        self.generation_count = 0

    async def record(self, cost: Optional[float]) -> None:
        if cost is None:
            return
        async with self._lock:
            self.generation_count += 1
            self.total_cost += cost
            logger.info(
                "[COST] generation #%d: $%.4f this call | $%.4f total this session",
                self.generation_count, cost, self.total_cost,
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "generation_count": self.generation_count,
        }

cost_tracker = CostTracker()

# --------------------------------------------------------------------------
# High-End Premium Prompt Construction
# --------------------------------------------------------------------------

DISPLAY_DIRECTIVES: Dict[str, str] = {
    "model": (
        "A professional female fashion model wearing this exact saree, draped elegantly in classic Nivi style. "
        "Full body visible, standing in a relaxed natural pose in a softly lit studio with a plain neutral backdrop."
    ),
    "mannequin": (
        "E-commerce ghost mannequin product photography. A hollow, invisible-neck dress form. "
        "Headless, limbless, empty inside. No human, no face, no skin. This exact saree is "
        "elegantly draped around the invisible plastic mannequin against a plain neutral studio backdrop."
    ),
    "flat": (
        "High-end, premium luxury e-commerce flat lay photography of the saree. The fabric is meticulously folded "
        "and laid perfectly flat on a clean, solid, pristine matte neutral studio surface. Shot from a precise "
        "overhead 90-degree top-down perspective with perfectly balanced, diffuse soft commercial studio lighting. "
        "No creases, no wrinkles, no humans, no models. The fabric texture, pallu, and intricate borders are showcased beautifully."
    ),
    "hanging": (
        "Premium luxury boutique display photography. The saree is elegantly draped and hung neatly on a minimalist "
        "high-end solid walnut wood clothes hanger, suspended against a clean, flawless matte neutral studio backdrop. "
        "Professional soft-box studio commercial lighting creating subtle, crisp dimensionality without harsh shadows. "
        "No humans, no models, completely empty background, showcasing the full length and natural drape of the luxury fabric perfectly."
    ),
}

def build_display_prompt(catalog: Dict[str, Any], display_type: str) -> str:
    directive = DISPLAY_DIRECTIVES.get(display_type, DISPLAY_DIRECTIVES["model"])
    fidelity_clause = (
        "This must be the EXACT SAME saree shown in the attached reference "
        f"image: same {catalog.get('primary_color', 'base')} base color, same "
        f"{catalog.get('secondary_color', 'border')} border color, identical "
        f"{catalog.get('motif', 'woven motif')} pattern and placement, identical "
        f"border width and design, and the same {catalog.get('fabric', 'fabric')} "
        "texture and sheen. Do not invent a new pattern, do not change any "
        "color, do not redesign the border or pallu."
    )
    style_notes = catalog.get("style_notes", "soft natural studio lighting, plain neutral backdrop")
    return f"{fidelity_clause} {directive} {style_notes}."

def _strip_data_uri(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value

def _as_data_uri(b64: str, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{b64}"

# --------------------------------------------------------------------------
# Runware client / Generation Core (Batch Routing)
# --------------------------------------------------------------------------

async def generate_images_batch(requests_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sends multiple generation tasks in a SINGLE API call for maximum speed."""
    if not RUNWARE_API_KEY:
        raise RunwareError("Runware API key missing.")

    tasks = []
    task_map = {} 

    for req in requests_data:
        display_type = req["display_type"]
        prompt = req["prompt"]
        reference_b64 = req["reference_b64"]

        if display_type == "model":
            target_model = "google:4@3" 
            extra_params = {"resolution": "1K"}
        else:
            target_model = "bfl:3@1" 
            extra_params = {
                "width": 832, 
                "height": 1248,
                "providerSettings": {"bfl": {"promptUpsampling": False}}
            }

        task_uuid = str(uuid.uuid4())
        task = {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "model": target_model, 
            "positivePrompt": prompt,
            "outputType": "URL",
            "includeCost": True,
            "inputs": {"referenceImages": [_as_data_uri(reference_b64)]}
        }
        task.update(extra_params)
        
        tasks.append(task)
        task_map[task_uuid] = {
            "display_type": display_type,
            "prompt_used": prompt,
            "backend_used": target_model
        }

    # Pack all tasks into one payload
    body = [{"taskType": "authentication", "apiKey": RUNWARE_API_KEY}] + tasks

    try:
        resp = await http_client.post(RUNWARE_URL, json=body, headers={"Content-Type": "application/json"})
    except httpx.HTTPError as exc:
        raise RunwareError(f"Could not reach Runware: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RunwareError(f"Runware returned a non-JSON response (HTTP {resp.status_code}).") from exc

    if resp.status_code >= 400 or payload.get("errors"):
        raise RunwareError(f"Runware error: {payload.get('errors', payload)}")

    results = []
    # Map the returning images back to their specific style using the UUID
    for item in payload.get("data", []):
        task_uuid = item.get("taskUUID")
        if task_uuid in task_map and ("imageURL" in item or "imageUUID" in item):
            meta = task_map[task_uuid]
            results.append({
                "image_url": item.get("imageURL"),
                "backend": meta["backend_used"],
                "display_type": meta["display_type"],
                "prompt_used": meta["prompt_used"],
                "cost": item.get("cost")
            })

    if not results:
        raise RunwareError("Runware returned no image data.")
        
    return results

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/cost-summary")
async def get_cost_summary():
    return JSONResponse(content=cost_tracker.summary())

@app.post("/api/analyze")
async def analyze_saree_inventory(file: UploadFile = File(...), style: str = Form(...)):
    if not gemini_client:
        return JSONResponse(status_code=500, content={"error": "Gemini API Key missing."})

    display_type = style if style in DISPLAY_DIRECTIVES else "model"

    try:
        img_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))

        extraction_prompt = """
        Analyze this saree image carefully and return ONLY raw JSON (no
        markdown fences, no commentary) with this exact shape:
        {
          "title": "Short, sellable product title",
          "primary_color": "the dominant base fabric color, as a plain color name",
          "secondary_color": "the dominant border/pallu color, as a plain color name",
          "fabric": "fabric type (e.g. silk, cotton, georgette, banarasi silk)",
          "motif": "short description of the woven/printed motif pattern (e.g. small gold booti, paisley zari border)",
          "description": "2-3 sentence luxury catalog description",
          "style_notes": "max 15 words, atmosphere/lighting only, e.g. 'soft golden-hour studio light, warm minimal backdrop'"
        }
        """

        response = await gemini_client.aio.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[extraction_prompt, pil_image],
        )

        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        catalog_payload = json.loads(clean_text)
        catalog_payload["image_prompt"] = build_display_prompt(catalog_payload, display_type)

        return JSONResponse(content=catalog_payload)

    except json.JSONDecodeError as error:
        logger.warning("Gemini returned non-JSON output: %s", error)
        return JSONResponse(status_code=502, content={"error": f"Gemini returned output that wasn't valid JSON: {error}"})
    except Exception as error:
        logger.exception("analyze_saree_inventory failed")
        return JSONResponse(status_code=500, content={"error": str(error)})

@app.post("/api/generate")
async def generate_model_images(req: GenerateRequest):
    clean_b64 = _strip_data_uri(req.reference_image)
    styles_to_process = req.display_types if req.display_types else ["model"]
    
    batch_requests = []
    
    for display_type in styles_to_process:
        if len(styles_to_process) == 1 and display_type == "model" and req.prompt_override:
            prompt = req.prompt_override
        else:
            prompt = build_display_prompt(req.catalog or {}, display_type)
            
        batch_requests.append({
            "display_type": display_type,
            "prompt": prompt,
            "reference_b64": clean_b64
        })
        
    try:
        # Trigger the optimized batch generation
        generated_results = await generate_images_batch(batch_requests)
        
        # Log costs synchronously after successful return
        for res in generated_results:
            await cost_tracker.record(res.get("cost"))
            
        return JSONResponse(content={"results": generated_results})
    except RunwareError as error:
        return JSONResponse(status_code=502, content={"error": str(error)})
    except Exception as error:
        logger.exception("Batch generation pipeline failed")
        return JSONResponse(status_code=500, content={"error": str(error)})