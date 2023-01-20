from .BiLstmModel import BiLstmModel
from .CnnModel import CnnModel
from .TransformerModel import TransformerModel

model_registry = {
    "cnn": CnnModel,
    "bilstm": BiLstmModel,
    "transformer": TransformerModel,
}
