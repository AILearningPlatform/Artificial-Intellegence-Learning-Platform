from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import asyncio
from starlette.requests import Request

app = FastAPI()

static_path = "/home/zkllmt/Documents/AI Section/Artificial-Intellegence-Learning-Platform/static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory="/home/zkllmt/Documents/AI Section/Artificial-Intellegence-Learning-Platform/templates")

API_KEY = "AIzaSyDHGZurLWlQlmBwypNz-hE8LEbCPgzhKnc"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("Community.html", {"request": request})

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted!")
    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"Received: {user_message}")

            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": user_message}]}]}

            response = requests.post(API_URL, headers=headers, json=data)
            if response.status_code == 200:
                reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                for word in reply.split():
                    await websocket.send_text(word + " ")
                    await asyncio.sleep(0.05)
            else:
                await websocket.send_text(f"Error: {response.status_code}")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()
