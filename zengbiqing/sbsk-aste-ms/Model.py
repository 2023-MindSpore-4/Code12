from Hyperparameters import Hyperparameter as hp
from ms_allennlp_fix.type_define import (
    InitializerApplicator
)
from ms_allennlp_fix.modules import (
    BasicTextFieldEmbedder, BertEmbedder
)
from ms_allennlp_fix.token_indexer_type import (
    PreTrainTransformerTokenIndexer
)

from span_model.models.span_model import SpanModel
from span_model.dataset_readers.span_model import SpanModelReader

def get_model(
        pretrain_trnsformer_indexer : PreTrainTransformerTokenIndexer,
        data_set_name: str = None
    ):
    params = hp.params

    token_embedders = {
        "bert" : BertEmbedder()
    }

    embedder = BasicTextFieldEmbedder(
        token_embedders
    )

    modules = params['model']['modules']

    init_alor = InitializerApplicator(
        regexes= params['model']['initializer']['regexes']
    )
    m_init_alor = InitializerApplicator(
        regexes= params['model']['module_initializer']['regexes']
    )

    sm = SpanModel(
        pretrain_transformer_indexers = pretrain_trnsformer_indexer,
        embedder=embedder,
        modules=modules,
        feature_size=params['model']['feature_size'],
        max_span_width=params['model']['max_span_width'],
        target_task=params['model']['target_task'],
        feedforward_params=params['model']['feedforward_params'],
        loss_weights=params['model']['loss_weights'],
        initializer= init_alor,
        module_initializer=m_init_alor,
        # regularizer=
        # display_metrics=
        use_ner_embeds=params['model']['use_ner_embeds'],
        span_extractor_type=params['model']['span_extractor_type'],
        use_double_mix_embedder=params['model']['use_double_mix_embedder'],
        relation_head_type=params['model']['relation_head_type'],
        use_span_width_embeds=params['model']['use_span_width_embeds'],
        use_bilstm_after_embedder=params['model']['use_bilstm_after_embedder'],
        # use_tag_embeds=
        # use_tree_embeds=
        # use_span_width=
        # use_dep=
        # dep_dim=
        # len_dep_type=
        # use_filter=,
        data_set_name= data_set_name
    )

    return sm