下面逐张图/表说明它们在做什么、看什么指标或分布（均出自你给的 PPT）。

1. Figure 4（Human 数据集特征分布）

* 内容：4 个直方图，依次为 Molecular Weight、LogP、NumHAcceptors、NumHDonors。
* 作用：展示**人体（Human）半衰期数据集**的基础理化性质分布，用于直观了解样本的“大小（MW）/疏水性（LogP）/受体/供体数量”等统计形态，判断是否偏斜、是否存在长尾，为后续建模与采样策略提供依据。

2. Figure 5（RAT 数据集特征分布）

* 内容：同样 4 个直方图，指标与 Figure 4 相同（MW、LogP、HBA、HBD）。
* 作用：展示**大鼠（RAT）半衰期数据集**的理化性质分布，便于与 Human 数据集对比：是否存在系统性分布差异（如 MW/LogP 的中心与方差不同），从而指导“合并训练 vs 物种分开训练”的选择。

3. Table 1（Model performance，半衰期>6h 的二分类）

* 内容：在 Human 与 Human+RAT 两种数据组合下，对比不同**不平衡处理策略**（无处理、SMOTE+TOMEK、Over Sampling）的分类效果。
* 指标：**ACC、Precision、Recall、F1、AUC**。
* 作用：量化不平衡学习策略对模型（XGBoost，6h 阈值）性能的影响，观察召回、AUC 是否随采样策略改善。

4. Figure 6（Confusion matrix，单折示例）

* 内容：二分类混淆矩阵（真实标签：short/long；预测标签：short/long）。
* 作用：直观看到**错分模式**与类别偏置——由于长半衰期样本更少，模型更容易预测为 **short**（类别不平衡导致）。

5. Figure 7（Feature importance，Top-50）

* 内容：XGBoost 的特征重要性条形图（前 50）。
* 作用：确认**决定性特征**主要与**疏水性（LogP/相关 VSA）**、**分子大小/体积**与**结构刚性**相关；与文献/既有研究对“口服稳定性决定因素”的认识一致。

6. Figure 8（SIF1 数据集表征）

* 内容：SIF1 的 4 个直方图：Molecular Weight、LogP、NumHAcceptors、NumHDonors。
* 结论注记：分布较集中、相较半衰期总数据集 **分子量更小**，但 **LogP 偏高**。
* 作用：给出**在小肠模拟液（SIF）数据集**上的分布特性，为阈值设定与模型分层训练提供数据直觉。

7. Figure 9（SIF2 数据集表征）

* 内容：同样 4 个直方图指标；分布更集中但**不均匀**，可见多峰/偏态；并指出**部分样本分子量过高、LogP 也偏高**；HBA/HBD 直方图呈多模态。
* 作用：提示 SIF2 的数据**分布漂移/多峰性**，暗示需要更稳健的验证或分布自适应策略。

8. Table 2（SIF/SGF 模型性能，对应阈值：SIF1/SGF1=6h；SIF2/SGF2=2h）

* 内容：分别在 **SIF1、SGF1、SIF2、SGF2** 四个数据集上训练**XGBoost 二分类**，输入特征为 Morgan 指纹、Avalon 指纹与 QED 特征；采用 5 折交叉验证。
* 指标：**ACC、Precision、Recall、F1、AUC**。
* 作用：系统比较不同模拟液数据集上的可分性与上限；一般 **SGF1/SIF1** 的指标更高，**SIF2/SGF2** 相对更难。

——以上即每幅图/表所衡量的对象、使用的指标与分布含义，均基于你提供的 PPT 内容整理。
