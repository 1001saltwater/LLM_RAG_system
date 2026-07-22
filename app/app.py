from fastapi import FastAPI
from app.routers.router_article import router as router_article
from app.routers.router_chunk import router as router_chunk
from app.routers.router_rag import router as router_rag
from app.routers.router_search import router as router_search


app = FastAPI()

app.include_router(router_article)
app.include_router(router_chunk)
app.include_router(router_rag)
app.include_router(router_search)