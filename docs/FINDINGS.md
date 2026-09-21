# Findings

*一个"只会做判断、不会写字"的模型（TypeSafe 的 Jev），能不能接管机械臂的任务级决策？
这是这台仿真实验台跑出来的全部结论。*

![all 20 scenes](media/reel.gif)

---

## TL;DR (English)

- **Bench**: MuJoCo + xArm7 pick-and-place. Each cycle a typed-judgment model (Jev: yes/no
  with probability, single choice with probabilities, graded score — no text output) picks
  the *intent* (approach / grasp / lift / carry / lower / release). Ordinary code owns the
  geometry, the safety veto and the workflow. No hardware involved.
- **It works**: 20/20 randomized scenes completed, median placement error 1.8 mm;
  grasp-state accuracy **100% on 79 unambiguous samples**, Brier **0.030** (0.25 = always
  answering 0.5). Any gate between **0.10 and 0.70** gives zero false positives and zero
  missed holds.
- **The measurement moved more than the model**: the same runs first scored 77% / Brier
  0.242. Two labelling fixes (an ambiguous "just closed, not yet proven" window, and
  capturing ground truth *before* the action instead of after) corrected 27 of 111 samples
  and took it to 100% / 0.030.
- **Fault injection** (8 single faults × 5–10 scenes, 70 scenes total): zero cases of
  confident-but-wrong grasp claims. A *lying* tactile sensor is survivable (100% with Jev vs
  0% for a hand-written rule); a **frozen camera** is not — it silently caps the
  task-progress estimate at 2.65/3, so the system never learns it is finished.
- **Fix, not a smarter model**: per-skill sensor-freshness gate + a stuck watchdog + escalate
  to a human. With it, the two blind-sensor cases stop and escalate 100% of the time, while
  the other six stressors keep completing with zero false alarms.
- **Limits**: quasi-static tabletop, hand-serialized state (**no real vision**), 5–10 scenes
  per fault, never touched hardware. (The grasp hold used to be a documented kinematic
  simplification; as of 2026-09-21 carrying is real contact friction — the "unresolved slip"
  turned out to be an IK side effect that teleported the arm on every move, see README —
  so slip faults are now expressible, though the failure map has not been re-run on the new
  physics yet.)

---

## 1. 实验台是什么

```
 真实世界 = MuJoCo（xArm7 + 桌面 + 方块 + 目标区）
        ↓  传感器层（带时间戳；相机 4 Hz / 触觉 20 Hz）
 状态：物体位置与高度、离目标距离、夹爪命令/实测间隙、触觉位、最近三次动作
        ↓  一次调用问三个问题（choice / noul / score）
 Jev：下一步用什么技能（9 选 1）· 抓稳没有（概率）· 任务到哪一步（0-3 分）
        ↓  代码否决权（门槛 + 依赖通道的新鲜度检查 + 卡死看门狗）
 技能层：approach / grasp / lift / carry / lower / release / retreat / hold / finish
```

状态里**故意不放**仿真真值：只有测量值、一个"指垫碰到东西"的触觉位、和动作历史。
所以"抓稳没有"必须由模型自己推断——也才能被评分。

## 2. 结果一：判断可靠性

20 个随机场景（方块位置 x∈[0.31,0.43]、y∈[0.06,0.20] + 小角度旋转），每场景 5–6 轮决策：

| 指标 | 数值 |
|---|---|
| 任务完成率 | 20/20 |
| 落点误差（中位 / 最差） | 1.8 mm / 2.1 mm |
| `grasp_secure` 准确率（79 个无歧义样本） | **100%** |
| Brier 分数 | **0.030**（"永远说 0.5" = 0.25） |
| 危险误判（说 ≥0.6 把握却没抓住） | **0** |

可靠性表（它说多少把握 → 实际发生多少）：

| 它说的把握 | 实际抓住 | 样本 |
|---|---|---|
| 0.0–0.2 | 0% | 39 |
| 0.6–0.8 | 100% | 33 |
| 0.8–1.0 | 100% | 7 |

阈值扫描：**门槛放 0.10–0.70 之间任意位置，假阳性 0、漏抓 0** —— 它对这个问题是决断的，
不是含糊的。但它还有一档 0.27–0.34 的回答，出现的时刻恰好是"刚夹上、还没抬起验证"的那几轮。

反面教材（很值得记住）：门槛是必须标定的参数。凭直觉设 0.55 时，24 轮决策被否决 20 轮，
机械臂一步不动——因为置信度公式是 (选项数×最高概率−1)/(选项数−1)，9 个选项时它再确定也只有 0.5 上下。

## 3. 结果二：测量比模型更容易出错（本项目最有价值的一条）

同一批模型行为，只改"怎么打分"：

| 轮次 | 我改了打分口径的什么 | 准确率 | Brier |
|---|---|---|---|
| 第 1 版 | ——（把"指垫有接触"就当抓住） | 77% | 0.242 |
| 第 2 版 | 剔除"刚合上、还没被抬起验证"的模糊状态 | 86% | 0.139 |
| 第 3 版 | 真值改为在**执行动作之前**采集 | **100%** | **0.030** |

被纠正的 27 条错判，模式惊人地整齐：

- **20 条全在 `grasp` 那一轮**：机械臂刚闭合手指、方块还没动过，模型答"只有 10% 把握"——
  这是**对的**（确实还没验证过），但我的标签说"有接触=抓住了"，于是判它错。
- **7 条全在 `release` 那一轮**：方块已放到目标区但**还没松开**，模型答"0.79 把握，抓着呢"——
  也是对的，但我的真值记在执行之后，那时已经松开了。

**结论**：看任何 AI 评测数字，第一个该问的不是"模型多强"，而是"标签怎么定的、真值什么时候采的"。
同一个模型可以因此从"看起来毫无信息量"变成"看起来接近完美"。

## 4. 结果三：失败地图

8 类单点故障，各跑 5–10 个场景（真实 API，共 70 场景 / 约 400 轮）。
除"抓取偏 22mm"是真物理外，其余都只污染**模型看到的信息**，世界保持真实，所以真值仍可信。

| 难例 | 完成率 | 升级给人 | 落点 mm | 危险FP | 误拒FN |
|---|---|---|---|---|---|
| none（基线） | 100% | 0% | 0.8 | 0 | 0 |
| 抓取标定偏 22 mm | 100% | 0% | **15.4** | 0 | 0 |
| 触觉永远说"没碰到" | 100% | 0% | 2.0 | 0 | 0 |
| 触觉永远说"碰到了" | 100% | 0% | 1.6 | 0 | 0 |
| 相机 4 秒一帧 | **0%** | **100%** | — | 0 | 0 |
| **相机冻结（拔线）** | **0%** | **100%** | — | 0 | 0 |
| 位置抖动 10 mm | 100% | 0% | 0.9 | 0 | 0 |
| 第 1 轮被外力推 | 100% | 0% | 1.4 | 0 | 0 |

三条发现：

1. **单个传感器失灵，它扛得住，而且强过手写规则。** 触觉全程报 False 的场景里，Jev 只看
   "夹爪间隙 87→59 mm + 物体离桌高度 + 最近动作"就判出抓住了（0.07 → 0.64 → 0.85），任务 100% 完成；
   同场景下手写规则替身死锁在"反复抓、永不抬"，完成率 0%。**状态里的冗余通道是鲁棒性的来源。**
2. **危险的不是"说谎的传感器"，是"冻结的传感器"。** 说谎会和别的通道矛盾，模型能交叉验证；
   冻结不矛盾、不报错、数据格式完全正常，只是永远不更新——所以它**永远不知道自己干完了**
   （`task_progress` 封顶 2.65/3，而只有 3 才会触发"完成"）。平均值看不出来，只有看最大值才发现。
3. **判断可靠 ≠ 物理精确。** 标定偏 22 mm 时判断依旧 100% 正确，但落点误差从 0.8 mm 涨到 15.4 mm。
   这类误差只能靠标定解决，判断层救不了，必须分开测。

### 解法：新鲜度闸门 + 卡死看门狗

`jev_arm/sensors.py` 让每条读数带 `age_s`；`main.py` 里每个技能声明自己依赖哪条通道，
超时（默认 1.0 仿真秒）就不许动、**拦住升级给人**；连续第三次要同一个动作且进度估计毫无变化时同理。
加装后：两个"盲传感器"场景 100% 安全停机，其余 6 个场景 **零误报**（含噪声场景），
而 `grasp_secure` 的准确率与 Brier 全部不变——闸门是代码层的补充，不动判断本身。

代价说清楚：闸门把"静默干不完"变成"停下来叫人"，**它不会让机械臂把活干完**，传感器还得人去修。

### 第二轮（2026-09-21 晚）：真实接触物理下的失败地图

物理从运动学携带改为真实摩擦（见第 5 节），并新增三个物理型难例（指垫打滑 μ=0.05 / 临界夹持
μ=0.08 / 工件 1.5 kg）。与上一轮**同种子 300–304**，唯一变量是物理；本轮用规则替身
（`--mode fake`）跑，因为要验证的是物理而不是判断模型。完整表见 README。

三条新结论：

1. **"抓取偏 22 mm"从精度问题升级为任务失败**：旧物理下它只是"落点偏 16.9 mm"，新物理下
   2/5 场景真的掉件——然后**重新抓取并完成**（落点 1.6 / 5.2 mm）。循环对掉件有恢复能力，
   而这个恢复过程过去没有任何指标记录过（新增的掉件率就是为它设计的）。
2. **完成判据本身会骗人**：`marginal_grip` 下方块在搬运途中滑落、恰好垂直落进目标区，
   被"在目标区 + 夹爪张开"判为完成（3/5 场景"完成"，落点中位 49.7 mm，刚好卡在 5 cm 半径内）。
   只有掉件率（100%）把它揭穿。这与第 3 节的教训同源——**判据和标签一样会骗人**，
   所以成功率必须和掉件率一起报。
3. **看门狗盲区（尚未修）**：`heavy_object` 下 grasp→lift→（方块滑回桌面）→grasp 交替
   20 轮不升级——触发条件只认"同一技能连续第三次"。候选改进：把周期为 2 的重复模式
   与"进度估计不变"组合起来触发。

传感器类难例（触觉两种、噪声、外力、相机两种）行为与上一轮一致，说明物理改动没有污染
传感层结论。

### 第三轮（2026-09-21 晚）：真实物理 + 真实判断（live）

同种子 300–304、同门控，11 类难例 × 5 场景 = 55 场景 / 571 轮决策，**全部由 Jev 判断**
（零 fallback），428 个可打分抓稳样本。完整表见 README。最重要的新发现：

**物体已经掉了，判断层还说握着。** `low_friction` 下危险FP = 12 轮：方块滑脱落回桌面后，
Jev 给出 0.60–0.67 的把握并提议 `carry`。它当时看到的是"命令间隙 = 实测间隙 = 42.2 mm
（指垫空合到底）+ 触觉 False + **方块高度 ≈ 0** + 最近动作 grasp/grasp/lift"——也就是说，
在"手指闭合、刚抓过"的状态下，它会**无视互相矛盾的冗余通道**，自信地搬运一只空手。
安全闸门也拦不住（0.62 > 门槛 0.45）。这是运动学携带时期不可能出现的失败模式，
也是真机上最贵的那一类：不报错、不升级，只是空手把动作演完。

其余三条：

- **触觉"永远说碰到了"从无害升级为危险**：60% 完成 / 60% 掉件 / 1 危险FP（旧物理下零误报），
  因为判断层相信触觉，滑脱后仍以为握着；
- **滑脱FP（新指标）零触发**——诚实记录：滑脱是瞬态，决策时刻物体要么还在手上、要么已无接触，
  真正抓住物理危险的是旧的危险FP；
- **看门狗盲区复现**：`heavy_object` 4/5 场景跑满 20 轮未升级（grasp↔lift 交替）。

旧结论在真实物理下依然成立：`tactile_dead` 100% 完成（靠间隙+高度+动作历史推断出因果关系）、
相机冻结/遮挡 100% 升级给人、`grasp_off` 判断依然准（97%）但物理上 40% 掉件——
**判断可靠 ≠ 物体不会掉**。

## 5. 局限

- 准静态桌面任务（不是飞行、不是动态环境、无人参与）；
- 状态是**手工序列化**的，没有真实视觉（感知前端被假定已完成）；
- 抓取后的"握住"曾是运动学携带的简化，所以"抓住后滑脱"这类接触失效当时无法表达；
  2026-09-21 已改为真实接触摩擦搬运（真值新增 `slipping` 旗标），滑脱故障已可注入，
  并在同种子下重跑了第二轮（规则替身，验证物理）与第三轮（live，同 Jev）失败地图
  （第 4 节末两小节）；
- 每种故障只有 5–10 个场景，够说明量级、不够当定论；
- 没上过真机。

## 6. 复现

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install mujoco numpy

# 规则替身，不需要 API key
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode fake --randomize --seed 1

# 真实 Jev（需要 TYPESAFE_API_KEY）
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode live --randomize --seed 1

# 批量 + 打分
.\.venv\Scripts\python.exe tools\batch_calibrate.py --mode live --n 20
.\.venv\Scripts\python.exe tools\summarize_log.py "logs/batch_live_20c/*.jsonl"

# 失败地图（先 --mode fake 免费验证，再 --mode live）
.\.venv\Scripts\python.exe tools\stress_batch.py --mode live --scenes 5
.\.venv\Scripts\python.exe tools\stress_map.py        # 从已有日志免费重算

# 把录制的决策重放成动画（零 API 花费）
.\.venv\Scripts\python.exe tools\replay_demo.py --logs "logs/batch_live_20c/*.jsonl"
```

## 7. 与相关工作的位置

| 工作 | 它做了什么 | 与这里的关系 |
|---|---|---|
| [jev-drone](https://github.com/RomanSlack/jev-drone)（96★, MIT） | 四旋翼用机载相机飞完 5 关障碍赛道；Jev 在 ~2.5 Hz 做战术判断，代码握否决权；只做"关掉 Jev"的消融 | 同一套思路的**动态任务**版本（难度更高、有真视觉）；但它没有对**模型自身判断**做量化，也没有故障注入 |
| [jev-mujoco](https://github.com/HiroRittsu/jev-mujoco)（无协议） | MuJoCo + xArm7 抓取放置；两级判断（意图 + 运动参数）；`JEV_MODE=fake` 可离线跑 | 最接近的实验台；它让模型也选运动参数，这里把几何全部留在代码里；它也没有可靠性量化 |
| [FARL](https://arxiv.org/abs/2601.07821)（清华 TEA Lab, 2026） | 世界模型预测"未来失败代价"，超过阈值就切到恢复策略；带 FailureBench（MetaWorld） | **形状相同**（阈值门控 + 切换），但它是风险门控、不是校准置信度，论文里没有阈值敏感性分析——这里的 0.10–0.70 扫描正好是那个空位 |
| [DeFog](https://arxiv.org/abs/2303.03391)（ICLR'23） | 离线 RL 容忍掉帧（训练时遮蔽 + 输入时间跨度） | 与"冻结相机"同一血脉，但**只在训练时容忍缺失观测，从不检测数据过期、也不升级给人** |
| [MoE-DP](https://arxiv.org/abs/2511.05007) | 专家模块对应语义原语，支持推理时重排子任务做错误恢复 | 与这层"任务级意图"最像的已发表工作 |
| [COPlanner](https://arxiv.org/abs/2310.07220)（ICLR'24） | 不确定性感知的 MPC | 该组唯一显式的不确定性机制，用作惩罚/探索奖励，不评分预测质量 |

一句话定位：**别人证明的是"这类模型能干成事"，这里量化的是"这个判断者有多可靠、在哪种故障下会悄悄失效"。**

## 8. 术语

- **Jev / System One**：TypeSafe 的类型化判断模型，只返回类型化答案与概率，不生成文字
  （[docs.typesafe.ai](https://docs.typesafe.ai/)）。
- **三个问题类型**：`noul`（是/否 → 概率）、`choice`（多选一 → 选中项 + 概率分布 + 置信度）、
  `score`（分级 → 加权分 + 概率分布 + 置信度）。
- **置信度 vs 概率**：概率是"每个选项各多大概率"，用来选最优；置信度由分布派生
  （n 个选项时 ≈ (n·峰值−1)/(n−1)），用来决定"自动执行还是交给人"。
- **危险FP / 误拒FN**：本项目给机器人场景定义的两个代价方向——"说抓住其实没抓住"（危险）
  与"能抓却不敢抓"（效率损失）。
