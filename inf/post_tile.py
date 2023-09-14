import requests
response = requests.post(
   url="http://localhost:8080/predictions/tsmodel",
   files={"data": ("3435-7280-14.npz", open("/home/tam/Documents/repos/time-series-for-lulc/data/cubesxy/3435-7280-14.npz", "rb"))}
)
print(response.content)
