# 底座切换决策（基于当前 fuser 负证据）

## 不应做的事

不要把现有 SD-1.4 GLIGEN checkpoint 直接替换为 SDXL、FLUX 或任意“更强” text-to-image checkpoint。GLIGEN 的 gated attention、PositionNet 和 UNet context/feature 尺寸与这些底座不兼容；强行加载既不会保留训练权重，也不能形成可比较的 VG 主结果。

当前服务器只发现一个生成底座：`gligen_checkpoints/diffusion_pytorch_model.bin`（SD-1.4/GLIGEN）。因此没有一个已验证、可直接切换的强模型 checkpoint。

## 文献上的候选与判断

1. **SATURN / VAR-CLIP（优先研究候选，不直接切换）**：SATURN 将 scene graph 排序为 token sequence，让冻结的 CLIP-VQ-VAE 与微调 VAR transformer 直接消费图结构。论文报告 VG 上很强的数值，但它是不同的自回归模型、不同训练程序，且本次查找未确认官方可复现实装/权重。它适合做“第二条完整基线”，不应混入当前 GLIGEN 协议。
2. **SGG-IG（最可复现的 diffusion 参考实现候选）**：其核心是 relation embedding 预训练 + spatial/image-scene alignment，再微调 SD-1.4；训练成本是 700k + 30k step。它更适合作为代码/损失设计参考，而不是在当前 AutoDL 余时内替换的轻量方案。
3. **工程上最现实的下一模型**：选择有开放权重的现代底座，但同时重建条件路径为 `scene graph -> cleaned caption + per-object region/layout map + relation-aware regional cross-attention/ControlNet`，而不是把 relation residual 加到 GLIGEN 的单一 object token。先做 10 张无训练/短训接线测试，再决定是否训练。

## 建议的下一章实验矩阵

固定不变：`standard_sg2im_fresh_h5/train.h5 -> test.h5`、共享 vocab、0 image-id overlap、当前固定 10 relation-supported 诊断集。

| 阶段 | 条件路径 | 目标 | 停止规则 |
|---|---|---|---|
| A | 当前 GLIGEN clean box/caption | 锚点 | 已完成 |
| B | GLIGEN triplet fuser | 验证弱注入假设 | 已否定：500 step 无可见改善 |
| C | 新底座 + region/layout control | 检验强空间注入是否可行 | 先固定 10；若主体或关系任一项下降则停止 |
| D | 新底座 + relation-aware regional attention | 只在 C 通过后加入 relation | 不与不同 sampler/split 的结果直接比较 |

阶段 C 的最小输入应是 4--6 个去重主对象的有序 region/map，caption 只承担全局颜色/风格；每条 relation 仅作用在对应的两个 region token，并在多尺度 attention/Control feature 中传播。这样避免当前“0.3% token residual”被强 UNet 去噪过程忽略。
