EventDeblur/
├── basicsr/                  # 您可以直接使用BasicSR的核心部分或修改它
│   ├── archs/               # 模型架构目录
│   ├── data/                # 数据处理目录
│   ├── losses/              # 损失函数目录
│   ├── metrics/             # 评估指标目录
│   ├── models/              # 模型训练/测试流程目录
│   ├── ops/                 # 自定义操作目录
│   └── utils/               # 工具函数目录
├── options/                  # 配置文件目录
│   ├── train/               # 训练配置
│   └── test/                # 测试配置
├── scripts/                  # 脚本目录
│   ├── train.sh             # 训练脚本
│   └── test.sh              # 测试脚本
├── tb_logger/                # TensorBoard日志
├── experiments/              # 实验结果目录
├── results/                  # 最终结果目录
├── docs/                     # 文档
├── README.md                 # 项目说明
├── requirements.txt          # 项目依赖
└── setup.py                  # 安装脚本
