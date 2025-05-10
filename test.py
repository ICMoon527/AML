import torch
from torchviz import make_dot
from Model import FullModel

# 创建模型实例（参数需与训练时一致）
model = FullModel(
    feature_dim=2,
    embed_size=32,
    num_layers=4,
    num_heads=4,
    device=torch.device("cuda"),  # 示例中使用 CPU
    forward_expansion=2,
    dropout=0.0,
    max_length=10000,  # 示例中缩短 max_length 避免内存爆炸
    seq_length=10000,
    num_classes=2,
    chunk_size=1000
)

# 创建示例输入（调整尺寸以适应 GPU 内存）
batch_size = 1
seq_length = 10000  # 缩短序列长度以简化计算图
x = torch.randn(batch_size, seq_length, 2)  # 输入形状 (batch, seq_len, feature_dim=2)

# 生成计算图
output = model(x)  # 前向传播
graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)

# 保存为文件（PDF 或 PNG）
graph.render(filename="Results/UMAP_Results/transformer_model_graph", format="pdf")  # 生成 PDF
# torch.onnx.export(model, x, "Results/UMAP_Results/transformer_model_graph.onnx")