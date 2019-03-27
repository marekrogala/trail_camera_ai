
FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl && \
 pip install torchvision && \
 pip install fastai

RUN pip install starlette uvicorn python-multipart aiohttp

COPY app.py .
COPY model.pkl .

EXPOSE 8008

CMD ["python", "app.py", "serve"]
