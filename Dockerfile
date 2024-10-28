#FROM python:3.9
FROM python:3.12-bullseye

# Define the working directory
RUN mkdir /app
RUN mkdir /results
#WORKDIR /app

COPY ./requirements.txt /requirements.txt

COPY ./app /app

COPY ./results /results

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# expose application to the 8000 port
EXPOSE 8000

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
#CMD ["uvicorn", "app.main.py", "--host=0.0.0.0", "--port=8000"] (Not working in Docker)