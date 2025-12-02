from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router

app = FastAPI(
    title="Speech Coach API",
    version="0.1.0",
)

# На MVP разрешаем всё — потом можно ужать CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(health_router)
app.include_router(analysis_router)
