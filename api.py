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

model_dir = "model"
model_path = os.path.join(model_dir, "model.h5")

#As model.h5 file too big, so i have uploaded in the drive and downloading here.
if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=184B4sZiOg23lW7MhfP11Moq2cuSTYvoz" 
    os.makedirs(model_dir, exist_ok=True)
    gdown.download(url, model_path, quiet=False)

# Loading the trained  model
model = load_model(model_path)



labels = ['cats', 'dogs', 'snakes']

# Create the fastapi app
app = FastAPI(title="Animal Image Classifier API")

#actual predition root
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and open the image using read()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))  # Resize to  dimenstion (224x224)

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  #  normalization 

        #  predictions
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = labels[predicted_index]
        confidence = float(np.max(predictions))

       
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

