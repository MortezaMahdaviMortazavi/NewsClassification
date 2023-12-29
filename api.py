import requests
import torch
import torch.nn.functional as F

from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

import pickle as pkl
import src.config as config

from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.models import RNN
from hazm import WordTokenizer
from dataclasses import dataclass

Config = {
    "CLASS_NAMES": [
        "Science and Culture",  
        "Economy",                
        "Social",                 
        "Politics",               
        "Sport",                  
        "Literature and Art",     
        "Miscellaneous"
    ]         
}


app = FastAPI()

class ModelInference:
    def __init__(self):
        vocab_path = 'vocabs/vocab_2.pkl'
        with open(vocab_path,'rb') as f:
            self.vocab = pkl.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = RNN(
                len(self.vocab),
                config.embed_size,
                config.embed_proj_size,
                config.hidden_size,
                config.is_bidirectional,
                config.num_layers,
                config.num_classes
        )
        self.classifier.load_state_dict(torch.load('checkpoints/model_checkpoint_2.pth'))
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device)

    def tokenizer(self,text):
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(text)
        tok2idx = [self.vocab.get(token,self.vocab['<UNK>']) for token in tokens]
        return torch.tensor(tok2idx).unsqueeze(0)
    
    def predict(self,text):
        encoded_text = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(self.classifier(encoded_text), dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return (
            Config["CLASS_NAMES"][predicted_class],
            confidence,
            dict(zip(Config["CLASS_NAMES"], probabilities)),
        )
    
def get_model():
    return ModelInference()

class NewsRequest(BaseModel):
    text: str


class NewsResponse(BaseModel):
    probabilities: Dict[str, float]
    news_class: str
    confidence: float

# @app.get("/predict")
def make_request():
    text_to_classify = "رژیم صهیونیستی در ادامه تجاوزات خود به خاک سوریه، بار دیگر اطراف دمشق را هدف قرار داد که در پی آن، چند نفر زخمی شدند."
    response = requests.get(f"http://127.0.0.1:8000/?text={text_to_classify}")
    if response.status_code == 200:
        result = response.json()
        print(result)
    else:
        print(f"Error: {response.status_code}, {response.text}")

# @app.get("/", response_class=FileResponse)
# async def read_item(request: Request):
#     return FileResponse("app/index.html", media_type="text/html")

# # Serve static files (like CSS or JavaScript)
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest, model: ModelInference = Depends(get_model)):
    news_class, confidence, probabilities = model.predict(request.text)
    return NewsResponse(
        news_class=news_class, confidence=confidence, probabilities=probabilities
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn main:app --reload
    # curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "your text here"}'
