
FROM python:3.7-slim
RUN mkdir -p /app
RUN chmod 777 /app
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py"]
