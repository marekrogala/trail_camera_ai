
FROM python:3.6-slim-stretch

RUN apt-get update
RUN apt-get install -y python3-dev gcc

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl && \
 pip install torchvision && \
 pip install fastai

RUN pip install starlette uvicorn python-multipart aiohttp aiofiles

COPY src /src

EXPOSE 8008

CMD ["python", "/src/app.py", "serve"]
