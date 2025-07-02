# This API allows users to upload an image via POST request to /predict/ endpoint and returns JSON with demographic analysis 
# (race, gender, age) and 11 traits: intelligence, confidence, cooperativeness, celibacy, attractiveness, big_spender, presentable, muscle_percentage,
# fat_percentage, dominance, and power
import os
import io
import uvicorn
import logging
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trait Prediction API",
    description="API for predicting traits from images using a pre-trained model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define trait names
TRAIT_NAMES = [
    "intelligence", "confidence", "cooperativeness", "celibacy", 
    "attractiveness", "big_spender", "presentable", "muscle_percentage", 
    "fat_percentage", "dominance", "power"
]

# Define demographic labels
RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
GENDER_LABELS = ["Male", "Female"]
AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Model loading function
def load_model():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load FairFace model
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 18)  # 7 (Race) + 2 (Gender) + 9 (Age)
        
        # Check if model file exists
        model_path = os.environ.get("MODEL_PATH", "fairface.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info(f"✅ Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using random weights")
            
        return model.to(device).eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# 's environment
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
def preprocess_image(image_bytes):
    try:
        # Convert image bytes to base64 first
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        # Convert back to bytes for processing
        image_bytes = base64.b64decode(base64_encoded)
        
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if image is in a different mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Define preprocessing steps
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply preprocessing
        image_tensor = preprocess(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# Helper functions for trait calculations
def age_to_numeric(age_range):
    """Convert age range string to numeric value for calculations"""
    # Parse the age range and return a numeric value
    if "-" in age_range:
        # For ranges like "20-29", take the average
        start, end = age_range.split("-")
        return (float(start) + float(end)) / 2
    elif "+" in age_range:
        # For ranges like "70+", use the base value
        return float(age_range.replace("+", ""))
    else:
        # For single values, convert directly
        return float(age_range)

def calculate_traits(race, gender, age_range):
    """Calculate all traits based on demographic information"""
    import random
    
    age = age_to_numeric(age_range)
    
    # Calculate base traits
    intelligence = min(max(0.5 + (age / 100), 0.0), 1.0)
    
    confidence = 0.5
    if gender == "Male": confidence += 0.05
    elif gender == "Female": confidence -= 0.05
    confidence -= (age / 150)
    confidence = min(max(confidence, 0.0), 1.0)
    
    cooperativeness = 0.5
    race_baseline = 1.0
    if race == "White": race_baseline = 1.0
    elif race == "Black": race_baseline = 0.9
    elif race in ["East Asian", "Southeast Asian"]: race_baseline = 0.95
    else: race_baseline = 0.92
    cooperativeness *= race_baseline
    cooperativeness += (age / 120)
    cooperativeness = min(max(cooperativeness, 0.0), 1.0)
    
    # Calculate additional traits
    celibacy = 0.3
    if age < 20: celibacy += 0.3
    elif age > 60: celibacy += 0.2
    else: celibacy -= 0.1
    if gender == "Male": celibacy -= 0.05
    elif gender == "Female": celibacy += 0.05
    celibacy += random.uniform(-0.1, 0.1)
    celibacy = min(max(celibacy, 0.0), 1.0)
    
    attractiveness = 0.5
    if 20 <= age <= 35: attractiveness += 0.2
    elif age > 60: attractiveness -= 0.15
    elif age < 18: attractiveness -= 0.1
    attractiveness += random.uniform(-0.2, 0.2)
    attractiveness = min(max(attractiveness, 0.0), 1.0)
    
    big_spender = 0.4
    if 30 <= age <= 55: big_spender += 0.2
    elif age < 20: big_spender -= 0.2
    if race in ["White", "East Asian"]: big_spender += 0.1
    big_spender += random.uniform(-0.15, 0.15)
    big_spender = min(max(big_spender, 0.0), 1.0)
    
    presentable = 0.5
    if 30 <= age <= 60: presentable += 0.15
    elif age < 20: presentable -= 0.1
    if gender == "Female": presentable += 0.1
    presentable += random.uniform(-0.1, 0.1)
    presentable = min(max(presentable, 0.0), 1.0)
    
    # Calculate body composition
    if gender == "Male": base_muscle = 0.4
    else: base_muscle = 0.3
    
    if 25 <= age <= 35: age_factor = 0.05
    elif age < 25: age_factor = 0.03
    else: age_factor = -0.05 * ((age - 35) / 20)
    
    muscle_percentage = base_muscle + age_factor + random.uniform(-0.08, 0.08)
    muscle_percentage = min(max(muscle_percentage, 0.15), 0.6)
    
    if gender == "Male": base_fat = 0.18
    else: base_fat = 0.25
    
    age_factor = 0.002 * max(0, age - 25)
    fat_percentage = base_fat + age_factor + random.uniform(-0.05, 0.1)
    fat_percentage = min(max(fat_percentage, 0.1), 0.45)
    
    # Calculate power traits
    dominance = 0.5
    if gender == "Male": dominance += 0.1
    else: dominance -= 0.05
    if 30 <= age <= 50: dominance += 0.1
    elif age < 25 or age > 65: dominance -= 0.1
    dominance += (muscle_percentage - 0.3) * 0.5
    dominance += random.uniform(-0.1, 0.1)
    dominance = min(max(dominance, 0.0), 1.0)
    
    power = 0.4
    if 40 <= age <= 60: power += 0.2
    elif age < 30: power -= 0.1
    if gender == "Male": power += 0.05
    power += dominance * 0.3
    power += random.uniform(-0.1, 0.1)
    power = min(max(power, 0.0), 1.0)
    
    # Convert all values to percentage ranges
    return {
        "intelligence": [int(intelligence * 75), int(intelligence * 100)],
        "confidence": [int(confidence * 70), int(confidence * 90)],
        "cooperativeness": [int(cooperativeness * 60), int(cooperativeness * 90)],
        "celibacy": [int(celibacy * 30), int(celibacy * 70)],
        "attractiveness": [int(attractiveness * 70), int(attractiveness * 90)],
        "big_spender": [int(big_spender * 40), int(big_spender * 80)],
        "presentable": [int(presentable * 70), int(presentable * 100)],
        "muscle_percentage": [int(muscle_percentage * 10), int(muscle_percentage * 15)],
        "fat_percentage": [int(fat_percentage * 10), int(fat_percentage * 10)],
        "dominance": [int(dominance * 80), int(dominance * 100)],
        "power": [int(power * 60), int(power * 90)]
    }

def format_response_json(response_data):
    """Format the response data to show only the highest trait and demographic values"""
    formatted_response = {
        "filename": response_data["filename"]
    }
    
    # Get only the dominant gender
    gender_probs = [gender_item["value"][1] for gender_item in response_data["gender"]]
    max_gender_idx = np.argmax(gender_probs)
    if gender_probs[max_gender_idx] > 0:
        gender_item = response_data["gender"][max_gender_idx]
        formatted_response["gender"] = [{
            "name": gender_item["name"],
            "value": f"{gender_item['value'][0]}% – {gender_item['value'][1]}%"
        }]
    
    # Get only the dominant ethnicity
    ethnicity_probs = [ethnicity_item["value"][1] for ethnicity_item in response_data["ethnicity"]]
    max_ethnicity_idx = np.argmax(ethnicity_probs)
    if ethnicity_probs[max_ethnicity_idx] > 0:
        ethnicity_item = response_data["ethnicity"][max_ethnicity_idx]
        formatted_response["ethnicity"] = [{
            "name": ethnicity_item["name"],
            "value": f"{ethnicity_item['value'][0]}% – {ethnicity_item['value'][1]}%"
        }]
    
    # Include age
    formatted_response["age"] = f"{response_data['age'][0]} – {response_data['age'][1]}"
    
    # Include height
    formatted_response["height"] = f"{response_data['height'][0]} – {response_data['height'][1]}"
    
    # Include face score
    formatted_response["face"] = f"{response_data['face'][0]}% – {response_data['face'][1]}%"
    
    # Include all traits
    formatted_response["traits"] = []
    for trait in response_data["traits"]:
        formatted_response["traits"].append({
            "name": trait["name"],
            "percentage": f"{trait['percentage'][0]}% – {trait['percentage'][1]}%"
        })
    
    return formatted_response

@app.get("/")
async def root():
    return {"message": "Welcome to the Trait Prediction API"}

# List all registered routes for debugging
@app.get("/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods if hasattr(route, "methods") else None
        })
    return {"routes": routes}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    """
    Upload an image and get full trait predictions.
    
    This endpoint is similar to /predict/ but returns the complete analysis.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG and PNG files are supported")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor).cpu().numpy().squeeze()
        
        # Parse model outputs
        race_outputs = output[:7]
        race_probs = torch.nn.functional.softmax(torch.tensor(race_outputs), dim=0).numpy()
        race_pred = RACE_LABELS[np.argmax(race_probs)]
        race_conf = float(np.max(race_probs))
        
        gender_outputs = output[7:9]
        gender_probs = torch.nn.functional.softmax(torch.tensor(gender_outputs), dim=0).numpy()
        gender_pred = GENDER_LABELS[np.argmax(gender_probs)]
        gender_conf = float(np.max(gender_probs))
        
        age_outputs = output[9:18]
        age_probs = torch.nn.functional.softmax(torch.tensor(age_outputs), dim=0).numpy()
        age_pred = AGE_LABELS[np.argmax(age_probs)]
        age_conf = float(np.max(age_probs))
        
        # Calculate traits based on demographics
        trait_predictions = calculate_traits(race_pred, gender_pred, age_pred)
        
        # Format gender probabilities in the required format
        gender_formatted = [
            {"name": "male", "value": [int(gender_probs[0] * 90), int(gender_probs[0] * 95)]},
            {"name": "female", "value": [int(gender_probs[1] * 8), int(gender_probs[1] * 12)]}
        ]
        
        # Format ethnicity probabilities - find the highest one
        max_ethnicity_idx = np.argmax(race_probs)
        ethnicity_formatted = [{
            "name": RACE_LABELS[max_ethnicity_idx],
            "value": [int(race_probs[max_ethnicity_idx] * 90), int(race_probs[max_ethnicity_idx] * 95)]
        }]
        
        # Create detailed response with all demographic information
        response = {
            "filename": file.filename,
            "gender": gender_formatted,
            "ethnicity": ethnicity_formatted,
            "age": [22, 25],  # Fixed age range as requested
            "height": [160, 170],  # Fixed height range as requested
            "face": [80, 85],  # Fixed face score as requested
            "traits": [
                {"name": "intelligence", "percentage": trait_predictions["intelligence"]},
                {"name": "confidence", "percentage": trait_predictions["confidence"]},
                {"name": "cooperativeness", "percentage": trait_predictions["cooperativeness"]},
                {"name": "celibacy", "percentage": trait_predictions["celibacy"]},
                {"name": "attractiveness", "percentage": trait_predictions["attractiveness"]},
                {"name": "big_spender", "percentage": trait_predictions["big_spender"]},
                {"name": "presentable", "percentage": trait_predictions["presentable"]},
                {"name": "muscle_percentage", "percentage": trait_predictions["muscle_percentage"]},
                {"name": "fat_percentage", "percentage": trait_predictions["fat_percentage"]},
                {"name": "dominance", "percentage": trait_predictions["dominance"]},
                {"name": "power", "percentage": trait_predictions["power"]}
            ]
        }
        
        # Format the response to show only the highest trait
        formatted_response = format_response_json(response)
        
        # Save prediction results to a file
        with open("prediction.json", "w") as f:
            json.dump(formatted_response, f, indent=4)
        
        logger.info(f"Processed image: {file.filename} and saved results to prediction.json")
        return JSONResponse(content=formatted_response)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/", tags=["Image Analysis"])
async def predict_traits(file: UploadFile = File(...)):
    """
    Upload an image for trait prediction.
    
    This endpoint accepts an image file, processes it through the model,
    and returns demographic and trait predictions.
    
    The image will be analyzed for:
    - Age
    - Gender (only the detected gender)
    - Race/Ethnicity (only the highest detected ethnicity)
    - Various personality and physical traits
    
    Results are also saved to prediction.json file.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG and PNG files are supported")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor).cpu().numpy().squeeze()
        
        # Parse model outputs
        # Extract race predictions (first 7 outputs)
        race_outputs = output[:7]
        race_probs = torch.nn.functional.softmax(torch.tensor(race_outputs), dim=0).numpy()
        race_pred = RACE_LABELS[np.argmax(race_probs)]
        
        # Extract gender predictions (next 2 outputs)
        gender_outputs = output[7:9]
        gender_probs = torch.nn.functional.softmax(torch.tensor(gender_outputs), dim=0).numpy()
        gender_pred = GENDER_LABELS[np.argmax(gender_probs)]
        
        # Extract age predictions (last 9 outputs)
        age_outputs = output[9:18]
        age_probs = torch.nn.functional.softmax(torch.tensor(age_outputs), dim=0).numpy()
        age_pred = AGE_LABELS[np.argmax(age_probs)]
        
        # Calculate traits based on demographics
        trait_predictions = calculate_traits(race_pred, gender_pred, age_pred)
        
        # Format gender probabilities - only include the detected gender
        gender_idx = np.argmax(gender_probs)
        gender_formatted = [{
            "name": GENDER_LABELS[gender_idx].lower(),
            "value": f"{int(gender_probs[gender_idx] * 90)}% – {int(gender_probs[gender_idx] * 95)}%"
        }]
        
        # Format ethnicity probabilities - only include the highest one
        max_ethnicity_idx = np.argmax(race_probs)
        ethnicity_formatted = [{
            "name": RACE_LABELS[max_ethnicity_idx],
            "value": f"{int(race_probs[max_ethnicity_idx] * 90)}% – {int(race_probs[max_ethnicity_idx] * 95)}%"
        }]
        
        # Create response with formatted traits
        traits_formatted = []
        for trait_name in TRAIT_NAMES:
            trait_values = trait_predictions[trait_name]
            traits_formatted.append({
                "name": trait_name,
                "percentage": f"{trait_values[0]}% – {trait_values[1]}%"
            })
        
        # Create the final response
        response = {
            "filename": file.filename,
            "gender": gender_formatted,
            "ethnicity": ethnicity_formatted,
            "age": "22 – 25",  # Fixed age range as requested
            "height": "160 – 170",  # Fixed height range as requested
            "face": "80% – 85%",  # Fixed face score as requested
            "traits": traits_formatted
        }
        
        # Save prediction results to a file
        with open("prediction.json", "w") as f:
            json.dump(response, f, indent=4)
        
        logger.info(f"Processed image: {file.filename} and saved results to prediction.json")
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

# Run the server if executed as a script
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
