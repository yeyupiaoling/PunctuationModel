import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieForTokenClassification


class ErnieLinear(nn.Layer):
    def __init__(self,
                 num_classes,
                 pretrained_token='ernie-3.0-medium-zh',
                 **kwargs):
        super(ErnieLinear, self).__init__()
        self.ernie = ErnieForTokenClassification.from_pretrained(
            pretrained_token, num_classes=num_classes, **kwargs)

        self.num_classes = self.ernie.num_classes
        self.softmax = nn.Softmax()

    def forward(self, input_ids, token_type_ids=None):
        y = self.ernie(input_ids, token_type_ids=token_type_ids)

        y = paddle.reshape(y, shape=[-1, self.num_classes])
        logits = self.softmax(y)

        return y, logits


class ErnieLinearExport(nn.Layer):
    def __init__(self,
                 num_classes,
                 pretrained_token='ernie-3.0-medium-zh',
                 **kwargs):
        super(ErnieLinearExport, self).__init__()
        self.ernie = ErnieForTokenClassification.from_pretrained(
            pretrained_token, num_classes=num_classes, **kwargs)

        self.num_classes = self.ernie.num_classes
        self.softmax = nn.Softmax()

    def forward(self, input_ids, token_type_ids=None):
        y = self.ernie(input_ids, token_type_ids=token_type_ids)

        y = paddle.reshape(y, shape=[-1, self.num_classes])
        logits = self.softmax(y)

        preds = paddle.argmax(logits, axis=-1)

        return preds
