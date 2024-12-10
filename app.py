from connections.more import *


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

@app.get("/Personalized_Learning_Paths", response_class=HTMLResponse)
async def PersonalizedLearningPaths(request: Request):
    return templates.TemplateResponse("personalized_learning_paths.html", {"request": request})

@app.get("/Hands_on_Projects", response_class=HTMLResponse)
async def HandsonProjects(request: Request):

    return templates.TemplateResponse("Hands_on_Projects.html", {"request": request})

@app.get("/Focus_Tools", response_class=HTMLResponse)
async def FocusTools(request: Request):
    return templates.TemplateResponse("Focus_Tools.html", {"request": request})

@app.get("/Progress_Tracking", response_class=HTMLResponse)
async def ProgresTracking(request: Request):
    return templates.TemplateResponse("Progress_Tracking.html", {"request": request})

#@app.post("/ImageClassification", response_class=HTMLResponse)
#async def Image_Classification(request: Request, Image: UploadFile = File(...)):
#    content = await Image.read()
#    nparr = np.frombuffer(content, np.uint8)
#    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#    img_name = f"static/uploaded/{Image.filename}"
#    cv2.imwrite(f"{img_name}", img)
#    return templates.TemplateResponse(
#        "Hands_on_Projects.html", {"request": request, "content": Image.filename, "image": img_name}
#    )
#
#    