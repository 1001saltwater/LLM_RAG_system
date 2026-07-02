from fastapi import FastAPI
from app.routers.router_article import router as router_article

app = FastAPI()

app.include_router(router_article)