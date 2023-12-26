### 使用训练好的模型进行推断

训练好的模型权重以及对应的一些代码文件在infer_files1文件夹中。

```bash
python infer.py
```

模型得到的推断生成结果会保存为5张图像，分别为：

- sample_app.png  采样表观隐变量的生成结果
- sample_pose.png  采样姿态隐变量的生成结果
- sample_head_app.png  仅采样头部表观隐变量的生成结果
- sample_left_upper_arm_pose.png  仅采样左上臂姿态隐变量的生成结果
- sample_left_lower_arm_pose.png  仅采样左下臂姿态隐变量的生成结果