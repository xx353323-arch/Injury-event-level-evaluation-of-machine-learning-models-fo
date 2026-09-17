# E10 数据流与各方法数据可及性（SoccerMon 事件级协议，10 seed 取值范围）

## 表 A 每个迁移任务的数据流

| Target | Source 有标注窗口（N/阳性） | Target support 有标注（阳性/阴性） | Target 无标注（本文 U_t）| Target test（阳性/阴性）|
|---|---|---|---|---|
| TeamA-2020 | 26019 / 104 | 87–218 / 348–872 | support 训练子集 290–728（不含 test）| 209–340 / 8018–8490 |
| TeamA-2021 | 25884 / 469 | 14–20 / 56–80 | support 训练子集 48–68（不含 test）| 42–48 / 9470–9502 |
| TeamB-2020 | 27431 / 496 | 7–19 / 28–76 | support 训练子集 24–64（不含 test）| 16–28 / 7982–8031 |

## 表 B 各方法可访问的数据成分

| 方法 | Source 有标注 | Target support 有标注 | Target 无标注 | 是否触及 test 特征 | 评估性质 |
|---|---|---|---|---|---|
| XGBoost | 全部 | 全部 support（与 source 拼接训练） | 无 | 否 | 归纳式 |
| LSTM | 全部 | 同上 | 无 | 否 | 归纳式 |
| GAT | 全部 | 同上 | 无 | 否 | 归纳式 |
| Transformer | 全部 | 同上 | 无 | 否 | 归纳式 |
| ProtoNet | 全部（episodic） | support 作原型 | 无 | 否 | 归纳式 |
| DANN（原稿实现） | 全部 | 全部 support | **test 集特征 X_te**（run_dann_10seed.py:145） | **是** | **直推式** |
| DANN（本次重跑） | 全部 | 全部 support | support 特征 | 否 | 归纳式 |
| 本框架 | 全部（episodic，源阳性仅 focal 与 MMD 用） | support 训练子集用于 MMD 与微调 | support 训练子集（U_t，仅取特征） | 否（paml_full_lmvg_v2.py:121,310-311） | 归纳式 |

说明：原稿 DANN 是唯一把测试特征当无标注域的方法；重跑后所有方法在同一 support 预算、同一 seed 集、同一事件级切分下比较。
