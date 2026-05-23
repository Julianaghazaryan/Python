from fastapi import FastAPI, HTTPException
from src.processor import run_pipeline
from src.model import PredictionResponse  # ՆՈՐ՝ Կապում ենք model.py-ն
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
app = FastAPI(
    title="High-Performance Student Data Analytics Engine",
    description="FastAPI microservice utilizing multiprocessing and machine learning."
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def get_ui():
    return FileResponse("static/index.html")
@app.get("/")
def read_root():
    return {"message": "Ուսանողների վերլուծության համակարգը միացված է"}


@app.post("/process", response_model=PredictionResponse)
def process_data():
    file_path = "data/Student_data.csv"
    
    try:
        analysis_results = run_pipeline(file_path)
        return {
            "status": "success",
            "results": analysis_results
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Ֆայլը չի գտնվել: Համոզվեք, որ ֆայլը տեղադրված է 'data/Student_data.csv' ճանապարհով:"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class PredictionInput(BaseModel):
    hours: float

@app.post("/predict")
def get_prediction(input_data: PredictionInput):
   
    slope = 0.0457
    intercept = 3.0659
    
 
    prediction = (input_data.hours * slope) + intercept
    
    return {"prediction": round(float(prediction), 2)}
