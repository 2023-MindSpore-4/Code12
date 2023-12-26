# RFM: response-aware feedback mechanism for background based conversation

The code for the paper [RFM: response-aware feedback mechanism for background based conversation](https://link.springer.com/article/10.1007/s10489-022-04056-4).



## Reference

If you use any source code included in this repo in your work, please cite the following paper.

```text
@article{chen2022rfm,
  title={RFM: response-aware feedback mechanism for background based conversation},
  author={Chen, Jiatao and Zeng, Biqing and Du, Zhibin and Deng, Huimin and Xu, Mayi and Gan, Zibang and Ding, Meirong},
  journal={Applied Intelligence},
  year={2022},
  publisher={Springer},
  doi={10.1007/s10489-022-04056-4}
}
```

## Requirements

- python 3.9/3.8/3.7
- mindspore 1.8.0

## Datasets

```

- As for Wizard of Wikipedia(WoW) dataset, we use the [DukeNet](https://github.com/ChuanMeng/DukeNet) version. Download the [Wizard of Wikipedia](https://drive.google.com/drive/folders/1zS0xRy-UgQTafNhxGBGS4in6zmAMKlVM) dataset, and put data files in the directory `dataset/wizard_of_wikipedia/`.
- Download the `glove.6B.300d.txt` and put it in `dataset/wizard_of_wikipedia/`.
- Furthermore,you should download nltk_data ,change the append path and put it {your_path}/RFM_MS/

```

```
To view the model, run in the cloud platform:


cd ${MA_WORKING_DIR}

python ${YOUR_PATH}/Run_RFM_WoW _testbatch.py 
--data_path=${YOUR_PATH}/dataset/wizard_of_wikipedia/ 
--model_path=${YOUR_PATH}/model/
```


