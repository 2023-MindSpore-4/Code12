# A Mindspore Implementation of Deformable_generator
### 环境配置

路径cd到项目文件夹下

```python
conda create -n mindspore python=3.9
conda activate mindspore

pip install -r requirements.txt
# 这个文件是直接用pip freeze导出来的
# 如果出了问题，按照代码里import的包一个一个装就可以
#mindspore我安装的是2.0.0-rc1版本，我的cuda版本是11.8
```

### 数据集

使用1Kfaces数据，分为train(900张)和eval（100张），放在data文件夹下就可以。数据文件夹结构为

```python
data
├── 1Kfaces
│   ├── eval
│   └── train
└── face_data.py
```

### 运行

训练中保存的图像，rec表示重构，sample表示采样生成的

```python
# 终端
python main.py --config configs/faces.yaml
# 重构和采样的图像以及模型权重全部保存在log文件夹里
```

### 评价、生成

这一步需要使用训练好的模型文件和保存的隐变量数据，这些都保存在log文件夹下，把路径复制，按照下面的命令格式执行即可。生成的图像在evaluate文件夹下

```python
python main.py --mode eval --config configs/faces.yaml --checkpoint 模型保存的权重.ckpt --latent 保存完的隐变量.pkl
```

**使用已有的训练好的模型，进行推断**

```
python main.py --mode eval --config configs/faces.yaml --checkpoint log/2023_08_02_12_21_04/dfg_model_epoch_999.ckpt --latent log/2023_08_02_12_21_04/latent_z.pkl
```

