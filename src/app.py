from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    open_image,
    get_transforms,
    models,
    load_learner
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
import base64

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

learn = load_learner("/src", fname="model.pkl")

def layout_response(body):
    return HTMLResponse(
        """
<html>
<head>
	<link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css">
	<script
	  src="https://code.jquery.com/jquery-3.1.1.min.js"
	  integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
	  crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.js"></script>

</head>
<body style="margin: 10px">
<h1 class="ui center aligned header">Wild boar or deer?</h1>
%s
</body>
</html>
    """ % body)


def ui_response(results, image_src):
    pred_class,pred_idx,outputs = results
    probs = dict(zip(learn.data.classes, map(float, outputs)))
    probs.pop(str(pred_class))
    others = ", ".join(["<b>%s</b> (%.2f%%)" % (k,v*100) for k,v in probs.items()])
    return layout_response("""

<div class="ui container">
<div class="ui card">
  <div class="image">
	<img alt="Embedded Image" src="%s" />
  </div>
  <div class="content">
    <a class="header">This is %s</a>
    <div class="meta">
      <span class="date">I'm %.2f %% confident about that</span>
    </div>
    <div class="description">
      Other possibilities are less likely: %s.
    </div>
  </div>
</div>
<div class="ui container">
<button class="ui button" onclick="history.go(-1)">Back</button>
</div>
</div>
""" % (image_src, pred_class, outputs[pred_idx]*100, others))


@app.route("/classify-upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes_as_json(bytes)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    content_type = data["file"].content_type
    bytes = await (data["file"].read())
    image_src = "data:%s;base64,%s" % (content_type, base64.b64encode(bytes).decode())
    return ui_response(predict_image_from_bytes(bytes), image_src)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return ui_response(predict_image_from_bytes(bytes), request.query_params["url"])

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    return learn.predict(img)

def predict_image_from_bytes_as_json(bytes):
    pred_class,pred_idx,outputs = predict_image_from_bytes(bytes)
    return JSONResponse({
        "predicted_class": str(pred_class),
        "class_probabilities": sorted(
            zip(learn.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return layout_response(
        """
<h2 class="ui center aligned header">Upload a photo taken in forest</h2>

<div class="ui container">
        <form action="/upload" method="post" enctype="multipart/form-data" class="ui form">
		<div class="field">
            	<label>Select image to upload:</label>
	        <input type="file" name="file">
		</div>
            <input type="submit" value="Detect wild animals" class="ui button primary"/>
        </form>

</div>

<h2 class="ui center aligned header">...or give me a URL of such image</h2>
<div class="ui container">
        <form action="/classify-url" method="get" class="ui form">
		<div class="field">
<label>Image URL:</label>
            <input type="url" name="url">
		</div>
            <input type="submit" value="Detect wild animals" class="ui button primary"/>
        </form>
</div>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
