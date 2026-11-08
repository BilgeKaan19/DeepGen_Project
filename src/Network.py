import torch
import timm

class LightNetwork(torch.nn.Module):
    def __init__(self, model_name='tinynet_a', pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name=model_name,
                                    pretrained=pretrained,
                                    num_classes=0,)

        self.fc = torch.nn.Linear(self.cnn.num_features, 512)
    def forward(self, input):
        feat = self.cnn(input)
        out = self.fc(feat)
        return out

    def get_embedding(self, input):
        """512-d face embedding (the output distilled from the ArcFace teacher)."""
        return self.forward(input)

    def get_features(self, input):
        """Backbone features before the final 512-d projection.
        (This is what `get_embedding` returned in the original code.)"""
        return self.cnn(input)
