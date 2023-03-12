



# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from load_dotenv import load_dotenv

# 2. Create the app object
load_dotenv()

origins = [
    "*",
    "*:*"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#. Load trained Pipeline
model = load_model(model_name = 'model-deploy', platform = 'aws', authentication = {'bucket' : 'titanic-bucket-deploy-v1'})

# Define predict function
@app.get('/')
async def index():
    return {'message':'Ping received'}


@app.get('/predict')
async def predict(PassengerId, Ticket, Name, Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked):
    data = pd.DataFrame([[PassengerId, Ticket, Name, Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked]], 
                        columns=['PassengerId', 'Ticket', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])

    predictions = predict_model(model, data=data) 
    return {'prediction': int(predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
