FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries and try to use docker cache by only copying the requirements
COPY requirements.txt ./requirements.txt
RUN pip install  -r requirements.txt

# https://docs.streamlit.io/deploy/tutorials/docker
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# set the working directory to the app
WORKDIR /app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]