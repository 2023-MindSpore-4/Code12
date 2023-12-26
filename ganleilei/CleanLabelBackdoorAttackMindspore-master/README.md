# CleanLabelBackdoorAttackMindspore

## Citation
This is the [MindSpore](https://www.mindspore.cn/en) implementation of [Triggerless Backdoor Attack for NLP Tasks with Clean Labels](https://aclanthology.org/2022.naacl-main.214/). If you find this repository helpful, please cite our paper.
```
@inproceedings{gan-etal-2022-triggerless,
    title = "Triggerless Backdoor Attack for {NLP} Tasks with Clean Labels",
    author = "Gan, Leilei  and
      Li, Jiwei  and
      Zhang, Tianwei  and
      Li, Xiaoya  and
      Meng, Yuxian  and
      Wu, Fei  and
      Yang, Yi  and
      Guo, Shangwei  and
      Fan, Chun",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.214",
    doi = "10.18653/v1/2022.naacl-main.214",
    pages = "2942--2952",
    abstract = "Backdoor attacks pose a new threat to NLP models. A standard strategy to construct poisoned data in backdoor attacks is to insert triggers (e.g., rare words) into selected sentences and alter the original label to a target label. This strategy comes with a severe flaw of being easily detected from both the trigger and the label perspectives: the trigger injected, which is usually a rare word, leads to an abnormal natural language expression, and thus can be easily detected by a defense model; the changed target label leads the example to be mistakenly labeled, and thus can be easily detected by manual inspections. To deal with this issue, in this paper, we propose a new strategy to perform textual backdoor attack which does not require an external trigger and the poisoned samples are correctly labeled. The core idea of the proposed strategy is to construct clean-labeled examples, whose labels are correct but can lead to test label changes when fused with the training set. To generate poisoned clean-labeled examples, we propose a sentence generation model based on the genetic algorithm to cater to the non-differentiable characteristic of text data. Extensive experiments demonstrate that the proposed attacking strategy is not only effective, but more importantly, hard to defend due to its triggerless and clean-labeled nature. Our work marks the first step towards developing triggerless attacking strategies in NLP.",
}
```
