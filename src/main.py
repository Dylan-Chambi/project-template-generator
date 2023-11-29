from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
from src.config import get_settings
from fastapi import FastAPI, Depends, File, UploadFile
from src.image_dect_service import ImageDectService


_SETTINGS = get_settings()

app = FastAPI(title=_SETTINGS.service_name, version=_SETTINGS.k_revision)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_service():
    return TemplateLLM()

def get_image_dect_service():
    return ImageDectService()

@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)

@app.post("/people_count")
def people_count(image_file: UploadFile = File(...), image_dect_service: ImageDectService = Depends(get_image_dect_service)):
    return image_dect_service.people_count(image_file)


@app.get("/")
def root():
    return {"status": "OK"}
