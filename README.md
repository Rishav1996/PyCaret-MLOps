 
# MLOps Using PyCaret
In this project we are demoing, how to use PyCaret module and deploy a model on AWS and Hugging Face

### Steps to setup the project

- Create an environment of python version 3.7 and download respective packages **PyCaret, Scikit-learn, boto3, fastapi, load-dotenv and pandas**
- Create a .env file as given below
~~~
AWS_ACCESS_KEY_ID = 'XXXXXX'
AWS_SECRET_ACCESS_KEY = 'XXXXXX'
AWS_ACCOUNT_ID = 'XXXXXX'
HUGGING_FACE_URL = https://{username}-{reponame}.hf.space/{route}
~~~  
- Trigger the notebooks in an order as mentioned below
~~~
Data Splitting Module -> Training & Deployment Module
~~~
- Upload the deploy module on the Hugging Face Space using Docker keep the repository public
- After the deploy module is running and up on the Hugging Face server continue with the final Notebook (Prediction Module.ipynb)
