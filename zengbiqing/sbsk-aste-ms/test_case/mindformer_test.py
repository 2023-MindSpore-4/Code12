from collections import OrderedDict
from mindformers import BertForPreTraining, BertConfig, BertTokenizer, AutoTokenizer, AutoProcessor, BertModel

def __show_supper_list(supper_list:OrderedDict):
    for key, values in supper_list.items():
        print(f"==={key}====>")
        for v in values:
            print(v)


def test_load_bert():
    import mindspore.ops as ops
    txt = ['test info [mask].']
    # txt = ['test', 'info', 'about', 'split', '.']
    max_seq_len = 512

    # bert_config = BertConfig(seq_length=512)
    # model = BertModel(bert_config)
    pretrain_conf =  BertConfig.from_pretrained('bert_base_uncased')
    pretrain_conf.__setattr__('seq_length', max_seq_len)
    model = BertModel(pretrain_conf)

    # model = BertForPreTraining.from_pretrained("bert_base_uncased")
    tokenizer = AutoTokenizer.from_pretrained("txtcls_bert_base_uncased")
    # process = AutoProcessor.from_pretrained("gpt2")

    tokens = tokenizer(txt, return_tensors='ms', max_length=512, padding='max_length')
    # tokens = process(txt)

    # input_ids = ops.unsqueeze(tokens['input_ids'], 0)
    # input_mask = ops.unsqueeze(tokens['attention_mask'], 0)
    # token_type_ids = ops.unsqueeze(tokens['token_type_ids'], 0)
    input_ids = tokens['input_ids']
    input_mask = tokens['attention_mask']
    token_type_ids = tokens['token_type_ids']

    print(input_ids)
    print(input_ids.shape, input_mask.shape, token_type_ids.shape)

    squence_embed, pooled_embed, embed_table = model(
        input_ids = input_ids,
        input_mask = input_mask,
        # token_type_id = token_type_ids
        token_type_ids = token_type_ids     # Bert Model
    )
    print(
        squence_embed,
        pooled_embed,
        embed_table
    )
    pass