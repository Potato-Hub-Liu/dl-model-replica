<<<<<<< HEAD
# Computer Vision Models PyTorch

<div align="center">
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**使用PyTorch从零实现的计算机视觉模型集合**

*从基础CNN到高级Vision Transformer的完整实现*

</div>

### 📖 项目简介

本项目致力于使用PyTorch的基础`nn.modules`从零实现经典和前沿的计算机视觉模型。通过渐进式的学习路径，帮助深度学习研究者和工程师深入理解各种模型架构的核心原理。

### ✨ 项目特色

- 🔧 **自底向上学习**: 从基础组件开始，逐步构建复杂模型
- 🛠️ **纯手工实现**: 手动实现Conv2D、BatchNorm、Attention等核心组件
- 📚 **渐进式学习**: 从基础组件到Stable Diffusion的完整学习路径
- 📝 **详细文档**: 每个模型都有完整的实现说明和理论背景
- 🧪 **可复现实验**: 提供完整的训练和评估脚本
- 🎯 **教育导向**: 注重代码可读性和教学价值

### 🗺️ 模型路线图

#### 第零阶段：PyTorch基础组件实现 (3-4周)
- [ ] **基础数学运算组件** - ReLU, Sigmoid, Tanh, Softmax
- [ ] **卷积和池化组件** - Conv2d, MaxPool2d, AvgPool2d
- [ ] **归一化组件** - BatchNorm2d, LayerNorm
- [ ] **线性层和正则化** - Linear, Dropout

#### 第一阶段：高级组件实现 (3-4周)
- [ ] **注意力机制组件** - MultiHeadAttention, 缩放点积注意力
- [ ] **循环神经网络组件** - LSTMCell, LSTM, BiLSTM
- [ ] **Transformer组件** - PositionalEncoding, TransformerBlock

#### 第二阶段：基础卷积神经网络 (2-3周)
- [x] **LeNet-5** (1998) - 使用自实现组件构建第一个CNN
- [ ] **AlexNet** (2012) - 引入现代技术(ReLU, Dropout)
- [ ] **VGG** (2014) - 深度网络设计和模块化

#### 第三阶段：残差网络与现代CNN (3-4周)
- [ ] **ResNet** (2015) - 残差连接的突破
- [ ] **ResNeXt** (2017) - 分组卷积的创新
- [ ] **DenseNet** (2017) - 密集连接的思想
- [ ] **EfficientNet** (2019) - 复合缩放的优化

#### 第四阶段：注意力机制与Transformer (4-5周)
- [ ] **Attention机制基础** - 使用自实现的注意力组件
- [ ] **Vision Transformer (ViT)** (2020) - Transformer在视觉的应用
- [ ] **Swin Transformer** (2021) - 层次化视觉Transformer
- [ ] **DeiT** (2021) - 数据高效的图像Transformer

#### 第五阶段：自监督学习模型 (3-4周)
- [ ] **Masked Autoencoder (MAE)** (2021) - 掩码自编码器
- [ ] **SimCLR** (2020) - 简单的对比学习框架
- [ ] **MoCo** (2019) - 动量对比学习

#### 第六阶段：高级与前沿模型 (4-6周)
- [ ] **ConvNeXt** (2022) - 现代化的卷积网络
- [ ] **CLIP** (2021) - 对比语言-图像预训练
- [ ] **DALL-E 2 组件** (2022) - 文本到图像生成
- [ ] **Stable Diffusion 组件** (2022) - 扩散模型核心组件

### ⏱️ 时间规划

- **第0阶段**: 3-4周 (PyTorch基础组件实现)
- **第1阶段**: 3-4周 (高级组件实现)
- **第2阶段**: 2-3周 (基础CNN)
- **第3阶段**: 3-4周 (ResNet等现代CNN)
- **第4阶段**: 4-5周 (Transformer系列)
- **第5阶段**: 3-4周 (自监督学习)
- **第6阶段**: 4-6周 (前沿模型)

**总计**: 22-30周 (约5.5-7.5个月)

### 🎯 学习路径特色

- **自底向上**: 从最基础的数学运算开始，逐步构建复杂模型
- **深度理解**: 手动实现每个组件，理解其数学原理和实现细节
- **模块化设计**: 先实现基础组件，再组合成完整模型
- **渐进式学习**: 从简单到复杂，循序渐进地掌握深度学习

<div align="center"></div>

**⭐ 如果这个项目对你有帮助，请给个星标支持！**

</div>