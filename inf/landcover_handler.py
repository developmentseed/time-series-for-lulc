from io import BytesIO
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler


class LandcoverHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.tfm = A.Compose(
        #     [
        #         A.Normalize(
        #             mean=[0.4601, 0.4658, 0.4240],
        #             std=[0.2548, 0.2543, 0.2850],
        #             max_pixel_value=255.0,
        #         ),
        #         # A.Resize(512,512), Resize is handled using PIL thumbnail to keep aspect ratio
        #         ToTensorV2(),
        #     ],
        # )

        # with open("_mlb.pkl", "rb") as f:
        #     self.mlb = pickle.load(f)

    def preprocess_one_image(self, req):
        print("Request", request.keys())
        im = req.get("data")
        if im is None:
            im = req.get("body")

        data = np.load(BytesIO(im), allow_pickle=True)
        X = data["X"]
        print("X Shape", X.shape)
        shape = X.shape
        X = X.reshape(-1, 10, 13).astype(np.float32)
        X = torch.from_numpy(X).to("cuda")
        return X

    def preprocess(self, requests):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaah")
        print(len(requests))
        print([r.keys() for r in requests])
        im = requests[0].get("data")
        if im is None:
            im = requests.get("body")

        data = np.load(BytesIO(im), allow_pickle=True)
        X = data["X"]
        print("XXXXXXXXXXXXXXXx", X.shape)
        shape = X.shape
        X = X.reshape(-1, 10, 13).astype(np.float32)
        X = torch.from_numpy(X).to("cuda")
        print("Great returning", X)
        return X
        # ims = [self.preprocess_one_image(req) for req in requests]
        # ims = torch.cat(ims)

        # return ims

    def inference(self, model_input):
        outputs = self.mo

    def postprocess(self, outputs):
        # res = list()

        # for o in outputs:
        #     o = torch.sigmoid(o).cpu()
        #     p = (o > 0.5).int().tolist()
        #     p = ",".join(l for l, i in zip(self.mlb.classes_, p) if i == 1)
        #     o = {l: i for l, i in zip(self.mlb.classes_, o.tolist())}
        #     res.append(
        #         {
        #             "category": p,
        #             "category_score": o,
        #         },
        #     )

        # return res
        logits = outputs
        print("outoutoautouto0", outputs)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        pred = pred.reshape(shape[:2])
        print(pred.shape)
        return pred
