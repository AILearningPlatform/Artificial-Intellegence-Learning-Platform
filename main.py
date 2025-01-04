from connections.more import *


GEMINI_API_KEY = "AIzaSyB1XEaD8Da6vqGeQZGZ5YbQzLUakDwlPoM"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

app = FastAPI()

model = Models()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})

@app.get("/courses", response_class=HTMLResponse)
async def features(request: Request):
    return templates.TemplateResponse("Courses.html", {"request": request})


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
async def handson_projects(request: Request, message: str = None, selected_model: str = None):
    return templates.TemplateResponse(
        "Hands_on_Projects.html",
        {
            "request": request,
            "message": message,
            "selected_model": selected_model,
        },
    )

@app.post("/choose_model")
async def choose_model(selected_model: str = Form(...)):
    return RedirectResponse(
        f"/Hands_on_Projects?selected_model={selected_model}", status_code=303
    )


@app.post("/upload_image")
async def upload_image(selected_model: str = Form(...), file: UploadFile = File(...)):
    if not file.filename:
        return RedirectResponse(f"/Hands_on_Projects?message= Upload an Image first!", status_code=303)
    file_path = f"./static/uploaded/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    if selected_model == "Mask R-CNN (Instance Segmentation)":
        result = model.Mask_R_CNN_Instance_Segmentation(file_path)
    elif selected_model == "ResNet-50 (Image Classification)":
        result = model.ResNet_50_Image_Classification(file_path)
    elif selected_model == "CycleGAN (Image-to-Image Translation)":
        result = model.CycleGAN_Image_to_Image_Translation(file_path)
    elif selected_model == "YOLOv11 (Object Detection)":
        result = model.YOLOv11_Object_Detection(file_path)
    elif selected_model == "Faster R-CNN (Object Detection)":
        result = model.Faster_R_CNN_Object_Detectio(file_path)
    elif selected_model == "VGG-16 (Image Classification)":
        result = model.VGG_16_Image_Classification(file_path)
    elif selected_model == "MobileNetV2 (Lightweight Image Classification)":
        result = model.MobileNetV2(file_path)
    else:
        result = "Please select a model"
    return RedirectResponse(f"/Hands_on_Projects?message={result[1]}&message2={result[0]}", status_code=303)

@app.get("/Focus_Tools", response_class=HTMLResponse)
async def FocusTools(request: Request):
    return templates.TemplateResponse("Focus_Tools.html", {"request": request})

@app.get("/Progress_Tracking", response_class=HTMLResponse)
async def ProgresTracking(request: Request):
    return templates.TemplateResponse("Progress_Tracking.html", {"request": request})

@app.get("/community", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("Community.html", {"request": request})

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted!")
    try:
        async with httpx.AsyncClient() as client: 
            while True:
                user_message = await websocket.receive_text()
                print(f"Received: {user_message}")

                headers = {"Content-Type": "application/json"}
                data = {"contents": [{"parts": [{"text": user_message}]}]}

                response = await client.post(GEMINI_API_URL, headers=headers, json=data)  
                if response.status_code == 200:
                    reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                    await websocket.send_text(reply)
                else:
                    await websocket.send_text(f"Error: {response.status_code}")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()

