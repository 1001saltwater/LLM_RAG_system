from fastapi import FastAPI
from app.routers.router_article import router as router_article
from app.routers.router_chunk import router as router_chunk


app = FastAPI()

app.include_router(router_article)
app.include_router(router_chunk)