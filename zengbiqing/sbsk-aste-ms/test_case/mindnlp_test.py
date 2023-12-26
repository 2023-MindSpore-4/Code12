from Hyperparameters import Hyperparameter as hp
import mindspore.dataset.text as text
from mindnlp.transforms.tokenizers import BertTokenizer

def test_Bert():
    word_list = ["test" ,"for", "voc", "load"]
    vob = text.Vocab.from_file(str(hp.vocabulary_path))
    bert = BertTokenizer(vob)

    test_text = "for test"
    print(bert(test_text))