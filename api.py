from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=184B4sZiOg23lW7MhfP11Moq2cuSTYvoz"  # replace with your own
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained CNN model
model = load_model(MODEL_PATH)


# Define class labels in the order used during training
class_labels = ['cats', 'dogs', 'snakes']

# Create the FastAPI app
app = FastAPI(title="Animal Image Classifier API")

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and open the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))  # Resize to match model input

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Same normalization as training

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions))

        # Return result
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

