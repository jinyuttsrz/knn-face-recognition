# 人脸识别系统 - Face Recognition System

适用于此项目的环境文件1.3.0.dist（python3.10）已上传至Releases

放置于python环境下{python310v（python环境目录）}\Lib\site-packages下即可

基于 K-近邻算法 (KNN) 的人脸识别系统，使用 `face_recognition` 库提取人脸特征。

## 功能特性
- 人脸检测与特征提取
- 基于 KNN 的人脸识别分类
- 实时结果显示识别框和姓名
- 支持自定义阈值调整识别精度
- 模型保存与加载功能

### 运行方式
- data/train 用于存放训练用数据集
- data/test  用于存放测试用数据集
- 运行`train_model.py`进行模型训练，模型训练完成后，会在data目录下生成knn_clf.pkl的文件
- 运行`recognize_faces.py`进行模型调用
