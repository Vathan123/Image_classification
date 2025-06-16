 Install Dependencies:-
           pip install -r requirements.txt 

           
How to Use the App:-
    Backend (FastAPI)
        Start the backend server with:
                  uvicorn api:app --host 0.0.0.0 --port 8000
        Access Swagger :-
                  docs: http://localhost:8000/docs
        Endpoint to upload image:
                  POST /predict
    Frontend (Flask)
        Start the frontend interface (after backend is running)
                  python app.py
        Open your browser and go to:
                  http://localhost:5000
    Upload an image and get prediction results (class + confidence)
