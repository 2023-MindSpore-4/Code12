- Token: 数据类，保存一个字符串的信息
- TokenIndexer: 将字符串转换为嵌入，默认实现 SingleIdTokenIndexer
    + SingleIdTokenIndexer: 
- Tokenizer: 将字符串切割为Token
- Field: 存储各类型的数据
- Instance: Field 的集合

- transformers.PreTrainedTokenizer._encode_plusa（由父类的encode_pluse调用
    + transformers.PreTrainedTokenizer.prepare_for_model