FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries and try to use docker cache by only copying the requirements
COPY requirements.txt ./requirements.txt
RUN pip install  -r requirements.txt

# set the working directory to the app
WORKDIR /app

CMD ["streamlit", "run", "app.py"]