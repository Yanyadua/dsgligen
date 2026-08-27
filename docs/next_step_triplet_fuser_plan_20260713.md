# VG → GLIGEN：endpoint-bound triplet fuser（下一步，不是已完成结果）

## 已知事实

- 固定 30 张的条件覆盖审计中，23/30 图的主要对象被 VG H5 支持，6/30 只部分支持，`2338873` 不支持（标注没有 woman/laptop）。因此后者不能用于语义控制结论。
- 在 10 张“对象 + 空间关系均人工核验支持”的固定子集上，关系 phrase token 确实改变了像素，但没有稳定地改善主体或关系；不能把它当作主方法。
- 旧 GAT 残差通过一个全局 gate 进入所有对象 token，先前 on/off 差异很弱。这不符合关系应只影响其 subject/object 的结构。

## 本次实现

`RelationTripletFuser` 以一个已声明的 `(subject, predicate, object)` 为单位，产生两份残差，仅 scatter-add 到该 subject 与 object token。它不新建一个全局 relation phrase token。

- 末层全零初始化：加载官方 GLIGEN 后，未训练时输出严格等于 base PositionNet 输出。
- 独立 sigmoid gate，初值 `sigmoid(-2)≈0.119`；不再被旧 GAT gate 压制。
- 残差上限为对应 base token norm 的 0.25。
- geometry 辅助预测将 relation geometry 清零后才送入 triplet fuser，避免以目标作为输入。

实现：`ldm/modules/diffusionmodules/scene_graph_grounding_net.py`。
未来训练模板：`configs/vg_standard_sg2im_triplet_fuser_clean_v1.yaml`。

## 已完成的零初始化接线检查

2026-07-13 使用固定的 10 个关系支持样本，`test.h5`、DDIM、50 steps、guidance 3.0、seed 20260713、clean caption/box、无 relation phrase token，生成了零初始化 triplet-fuser 输出。它与已有 base clean 输出逐文件 SHA-256 和逐像素 RGB 完全一致：`10/10` byte-identical、mean MAE `0.0`、max absolute difference `0`。

输出目录：`eval_outputs/clean10_triplet_zero_c_no_token_retry4_20260713`（远程），本地副本在 `artifacts/clean10_triplet_zero_smoke_20260713/`。这只证明加载与旁路兼容，不证明 fuser 已有控制效果。

## 受控短训结果：否定性诊断

2026-07-13 完成了一个独立的 500-step 小预算训练：`VG train.h5`（62,565 image IDs）→ `VG test.h5`（5,096 image IDs），overlap `0`；179 object / 46 predicate vocab；DDIM 50、guidance 3.0、seed 20260713 用于固定 10 张诊断。只有 13 个参数张量可训练（triplet fuser、triplet gate、relation-geometry predictor），没有 SD/UNet/GAT/fuser 参数混入。checkpoint 分别保存于 1/101/201/301/401/500 step。

10-step smoke 已确认 triplet final layer 的 norm 从 `0` 变为 `0.0501`，所以分支确实获得梯度。500-step checkpoint 中 final layer norm 为 `4.3817`，并成功被 eval 加载（13 tensors）。但 10 张输出相对 clean base 的平均 RGB MAE 仅 `0.156/255`，肉眼没有稳定的主体或空间关系改善；因此它不是 clean main result，也不应继续延长训练。

真实 token 量级解释了该现象：base token norm `46.99`，triplet delta norm `1.238`，学习到的 gate sigmoid `0.117`，实际 contribution norm 只有 `0.145`（base 的约 `0.31%`）。为排除仅是 gate 过小，额外做了明确标注为 ablation 的 inference-only gate override：logit `0` / sigmoid `0.5`。它把平均像素 MAE 提至 `0.534/255`，仍未显示可用的语义或关系控制。因此当前结论是：**endpoint-bound fuser 的数据/训练接线正确，但 GLIGEN 当前 object-token 注入位点的控制杠杆不足；不要继续堆这条分支的训练步数。**

对应远程产物：

- `OUTPUT_STANDARD_SG2IM_TRIPLET_FUSER/vg_standard_sg2im_triplet_fuser_500step_20260713/tag00/checkpoint_00000500.pth`；
- `eval_outputs/clean10_triplet_fuser500_clean_c_no_token_20260713`；
- `eval_outputs/clean10_triplet_fuser500_clean_primary_c_no_token_20260713`；
- `eval_outputs/clean10_triplet_fuser500_gate0_ablation_retry_20260713`（ablation only）。

## 协议（若且仅若获得训练批准）

训练源固定为 `/root/autodl-tmp/standard_sg2im_fresh_h5/train.h5`；评测源固定为同目录 `test.h5`；二者共享同一 `vocab.json`。模板使用 `clean_spatial_v1`、`clean_primary`、最多 6 个对象和 1 条关系。它不启用 relation grounding phrase token。

训练时冻结 SD/UNet、GLIGEN fusers 和 PositionNet 原始 object/box MLP，只开放：

1. `triplet_fuser`；
2. `triplet_gate`；
3. `relation_geo_predictor`。

初始损失仅为 diffusion + `0.05 × masked relation-geometry prediction`。object align、spatial consistency、graph-image align、graph distillation、masked graph pretraining 都关闭。

## 允许短训前的门槛

先以相同 seed、DDIM 50 steps、guidance 3.0 在固定 10 张关系支持样本上比较：

- base clean caption / box；
- 当前 relation-token D（scale 0.5）；
- triplet-fuser（零初始化，应逐像素等同 base）；
- 若短训获批后的 triplet-fuser。

人工盲评至少分别记录主体出现、关系方向、颜色/写实感、明显伪影；必须逐张保存 condition metadata。只有在主体与关系两项都不差于 base、并有重复 seed 支持时，才扩大到固定 30 张。此前不计算或比较 FID/IS/OOR。

## 回退

若短训后出现主体漂移、颜色/画质下降，或受支持关系没有一致改善：将 `use_triplet_fuser: false`（或加载零初始化）立即回到当前 clean box/caption baseline；保留审计、种子、配置和 checkpoint，不将该 checkpoint 作为 clean main result。
