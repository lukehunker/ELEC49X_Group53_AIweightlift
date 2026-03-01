#!/usr/bin/env python3
"""
FastAPI Server for RPE Prediction

Handles video uploads from Flutter app and returns RPE predictions.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /predict - Upload video, get RPE prediction
    GET /health - Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import lightgbm as lgb
import joblib
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from OpenFace.openface_inference import extract_features as extract_openface
import Bar_Tracking.barspeed_demo as barspeed
import MMPose.feature_extraction_deadlift as mmpose_deadlift
# Import other feature extractors as needed

app = FastAPI(
    title="AI Weightlift Coach API",
    description="RPE prediction from workout videos",
    version="1.0.0"
)

# CORS - Allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
MODEL_DIR = Path(__file__).parent / "Train_Outputs"
MODEL = None
ENCODER = None

@app.on_event("startup")
async def load_models():
    """Load LGBM model and label encoder on server start"""
    global MODEL, ENCODER
    
    model_path = MODEL_DIR / "lgb_rpe_predictor.txt"
    encoder_path = MODEL_DIR / "lift_type_encoder.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    
    MODEL = lgb.Booster(model_file=str(model_path))
    ENCODER = joblib.load(encoder_path)
    
    print("✅ Models loaded successfully!")
    print(f"   Lift types: {list(ENCODER.classes_)}")


@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "AI Weightlift Coach API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST with video)",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "encoder_loaded": ENCODER is not None
    }


@app.post("/predict")
async def predict_rpe(
    video: UploadFile = File(...),
    lift_type: str = Form(...),
    expected_reps: Optional[int] = Form(None)
):
    """
    Predict RPE from uploaded workout video
    
    Args:
        video: Video file (mp4, mov)
        lift_type: Type of lift (Bench Press, Squat, Deadlift)
        expected_reps: Number of reps performed (optional, for validation)
    
    Returns:
        {
            "predicted_rpe": 8.5,
            "confidence": 0.92,
            "lift_type": "Squat",
            "features": {
                "openface_detection_rate": 0.95,
                "bar_speed_avg": 0.45,
                ...
            },
            "warnings": []
        }
    """
    
    # Validate lift type
    if lift_type not in ENCODER.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lift_type. Must be one of: {list(ENCODER.classes_)}"
        )
    
    # Validate video format
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please use MP4, MOV, or AVI"
        )
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    video_path = None
    
    try:
        # Save uploaded video to temp file
        video_path = os.path.join(temp_dir, f"upload_{video.filename}")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        print(f"Processing video: {video.filename} ({lift_type})")
        
        # Initialize feature dict
        features = {}
        warnings = []
        
        # 1. Extract OpenFace features (facial expression)
        print("  [1/3] Extracting facial features...")
        try:
            openface_features = extract_openface(video_path, verbose=False)
            if 'error' in openface_features:
                warnings.append(f"OpenFace failed: {openface_features['error']}")
            else:
                features.update(openface_features)
                
                # Check detection quality
                detection_rate = openface_features.get('detection_rate', 0)
                if detection_rate < 0.5:
                    warnings.append(
                        f"Low face detection rate ({detection_rate*100:.0f}%). "
                        "Ensure face is visible in video."
                    )
        except Exception as e:
            warnings.append(f"OpenFace extraction failed: {e}")
        
        # 2. Extract MMPose features (body keypoints)
        print("  [2/3] Extracting body pose features...")
        try:
            # TODO: Implement pose extraction for each lift type
            # For now, using placeholder
            # mmpose_features = extract_mmpose(video_path, lift_type)
            # features.update(mmpose_features)
            pass
        except Exception as e:
            warnings.append(f"Pose extraction failed: {e}")
        
        # 3. Extract bar speed features
        print("  [3/3] Tracking bar speed...")
        try:
            # TODO: Implement bar tracking
            # bar_features = extract_bar_speed(video_path)
            # features.update(bar_features)
            pass
        except Exception as e:
            warnings.append(f"Bar tracking failed: {e}")
        
        # Encode lift type
        lift_encoded = ENCODER.transform([lift_type])[0]
        features['lift_type_encoded'] = lift_encoded
        
        # Check if we have minimum features for prediction
        if len(features) < 10:  # Adjust threshold based on your model
            raise HTTPException(
                status_code=422,
                detail="Insufficient features extracted from video"
            )
        
        # Prepare features for model (match training feature order)
        # TODO: Create proper feature vector matching your training
        feature_vector = [features.get(key, 0) for key in MODEL.feature_name()]
        
        # Predict RPE
        print("  Making prediction...")
        predicted_rpe = MODEL.predict([feature_vector])[0]
        
        # Calculate confidence (simplified - you can improve this)
        confidence = min(features.get('detection_rate', 0.8), 1.0)
        
        result = {
            "predicted_rpe": round(float(predicted_rpe), 1),
            "confidence": round(float(confidence), 2),
            "lift_type": lift_type,
            "video_name": video.filename,
            "features": {
                "detection_rate": features.get('detection_rate', 0),
                "total_frames": features.get('total_frames', 0),
                # Add other key features for transparency
            },
            "warnings": warnings if warnings else None
        }
        
        print(f"  ✅ Prediction: RPE {result['predicted_rpe']}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temp files
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/batch_predict")
async def batch_predict(videos: list[UploadFile] = File(...)):
    """
    Process multiple videos at once (for future use)
    """
    results = []
    for video in videos:
        # Process each video
        pass
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
