# 基于表面肌电信号的动作识别（深度学习）

# 1、sEMG的基础知识
## 1-1 sEMG的产生

表面肌电信号是由**多个运动单元**发放的**动作电位序列**，在皮肤表面呈现的**时间上和空间上**综合叠加的结果。

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/sEMG_generation.png" width=60% height=60% />  
</div>
<div align="center"> 图1 肌电信号生成</div>

**sEMG的特点：**
* 幅值一般和肌肉运动力度成正比，能精确的反映肌肉自主收缩力
* 超前于人体运动30-150ms产生

## 1-2 基于sEMG的动作识别一般处理流程

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/sEMG_ML_process.png" width=60% height=60% />  
</div>
<div align="center"> 图2 基于机器学习的肌电识别处理流程</div>

（1）离线采集sEMG
* 定义动作数量、动作类型
* 选择采集设备：Delsys(2000Hz)、Myo(200Hz)、OttoBock(100Hz)、高密度阵列式等
* 肌肉位置的选择、电极数量的选择：根据肌肉解剖位置调整电极
* 引导方式：图片、语音
* 采集流程：休息+动作循环采集
* 休息时间、动作时间，动作维持的力的大小，动作的姿势尽量保持一致

（2）数据预处理
* 10-350Hz带通滤波器，50Hz陷波器
* 标签修正：数据裁剪、最大面积法、极大似然修正
* 样本不均衡问题：休息动作的处理（通过阈值）
* 特征归一化：min-max标准化、标准差归一化
* 数据增强：加高斯噪声、翻转信号通道、时间窗+增量窗

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/label_corrected.png" width=90% height=90% />  
</div>
<div align="center"> 图3 基于最大面积法的动作标签修正(红线表示未修正的标签，黑线表示修正后的标签)</div>

.

（3）特征提取：时域、频域、时频域（tsfresh库）

（4）特征选择：
* 过滤法：方差选择法、相关系数法、卡方检验、互信息法，评估单个特征和结果值之间的相关程度，排序留下Top相关的特征部分
* 包裹型：递归特征删除法、基于学习模型的特征排序
* 嵌入型：正则化方法（L1正则化筛选特征）

（5）特征降维：
* PCA、LDA、SVD分解、流行学习LLE（非线性降维）、自编码器、T-SNE

（6）模型训练：
* KNN、LDA、DT、LR、NB、SVM、ANN
* RF、AdaBoost、GBDT、LightGBM、XGBoost
* AE、MLP、深层玻尔兹曼机、深层信念网络、CNN、RNN、LSTM、Inception、Attention
* 迁移学习、GAN

（7）在线控制决策：
* 预测结果做平滑处理，1s判别10次，3-5次投票作为控制指令
* 机器人正在运动时不接受指令
* 机器人闭手状态时只接受开手类指令（康复机器人）

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/sEMG_control.png" width=60% height=60% />  
</div>
<div align="center"> 图4 肌电控制手部康复机器人</div>

## 1-3 基于深度学习的处理流程：实现端到端的动作识别

（1）离线采集sEMG，并数据预处理

（2）构造肌电图像，输入给深度学习模型

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/sEMG_DL_process.png" width=60% height=60% />  
</div>
<div align="center"> 图5 基于深度学习的肌电识别处理流程</div>

# 2、数据集 

## 2-1 NinaPro 数据集

[NinaWeb](http://ninapro.hevs.ch/node/7)

[NinaPro数据下载及数据说明](https://datadryad.org//resource/doi:10.5061/dryad.1k84r)

NinaPro DB1: OttoBock(100Hz)采集设备，粘贴10个电极，27位健康受试者，53种手部动作（包含休息状态）

NinaPro DB2：Delsys(2000Hz)采集设备，粘贴12个电极，40位健康受试者，49种手部动作（不包含休息状态）

## 2-2 Medical & Rehabilitation Robot Laboratory of SIA 数据集

[SIA_delsys_16_movements_data数据下载](https://download.csdn.net/download/malele4th/11088765)

SIA_delsys_16_movements数据集：Delsys(2000Hz)采集设备，粘贴6个电极，4位健康受试者，16种手部动作

6个电极的粘贴位置：前臂的桡侧腕短伸肌、桡侧腕屈肌、肱桡肌、尺侧腕伸肌、指伸肌、指浅屈肌

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/SIA_delsys_16_hand_movements.png" width=60% height=60% />  
</div>
<div align="center"> 图6 16手部动作</div>

# 3、方法

## 3-1 传统机器学习方法

每个通道提取多个特征，RMS、MAV、WL、ZC、SSC等

## 3-2 深度学习方法

（1）NinaPro DB1：输入图像大小 12 * 10 （120ms * 10channels）

（2）NinaPro DB2: 输入图像大小 200 * 12 （100ms * 12channels）

（3）SIA_delsys：输入图像大小 200 * 6 （100ms * 6channels）

## 3-3 网络结构

使用Conv1D、Conv2D、Alternate-CNN（交替卷积）、ML-CNN（多流卷积操作+大池化层）四种

#### NinaPro DB1中的ML-CNN类似于NLP中的TextCNN模型，没有Embedding层

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/NinaPro-DB1-TextCNN.png" width=60% height=60% />  
</div>
<div align="center"> 图7 应用于肌电识别的TextCNN模型</div>

#### ML-CNN（Multi-stream convolutional operation and large pooling window CNN）

<div align="center">
<img src="https://github.com/malele4th/sEMG_DeepLearning/blob/master/picture/SIA_delsys_MLCNN.png" width=80% height=80% />  
</div>
<div align="center"> 图8 应用于肌电识别的ML-CNN模型</div>




