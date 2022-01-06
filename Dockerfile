FROM python:3.9-buster
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN curl https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth -o /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth
RUN gdown --fuzzy https://drive.google.com/file/d/1BBkYeUC8rj1lzy0F8iG6ONaxkSvO6YDS/view?usp=sharing
RUN gdown --fuzzy https://drive.google.com/file/d/1-2QxKk0w8mqwUSXoQx6AJB99pqSNdoVF/view?usp=sharing 
CMD ["python3","run.py"]
