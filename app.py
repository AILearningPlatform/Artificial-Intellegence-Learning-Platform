from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})

@app.get("/courses", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("Courses.html", {"request": request})

@app.get("/community", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("Community.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("About.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("Contact.html", {"request": request})

