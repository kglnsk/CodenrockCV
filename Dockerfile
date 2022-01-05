FROM python:3.9-buster
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN curl https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth -o /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth
CMD ["python3","run.py"]
