"""FastAPI application entrypoint."""

from fastapi import FastAPI

from api.deps import init_state
from api.routes import router


app = FastAPI(title="Credit Scoring API", version="0.1.0")
app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    init_state()
