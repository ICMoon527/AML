# AML
    1. 各管文件分别处理，第一管与第一管对齐，第二管与第二管对齐...  √
    2. 各管文件的数据集合起来处理，相互映证。
        结果：不同病人之间的差异性大。
    3. 不同病人（相同亚型）的可以打乱混在一起做分类训练，看是否能提高测试集的表现。
        结果：貌似可以，但还是不够好
    4. 加入病人信息与不加入做对比
    5. 加注意力机制作为权重参考与不加入的结果做对比
    6. 考虑到cd19和cd56经常一起用，置0的时候也同时置0作为一个测试
    7. Dropout Layer放在激活层之后  √
    8. data augmentation
    10. 5-fold cross-validation scheme
    11. AUC: https://zhuanlan.zhihu.com/p/81202617
    12. PPS(Predictive Power Score)被提出用来挖掘特征之间的关系。
    13. 把比较冗余的参数去掉重新训练，看能否提高泛化能力。
    14. 冗余的去掉以后，看能否提高不shuffle的情况下，模型对测试集的表现。

## 数据处理
### 数据获取
    找M2,M4,M5的病人，
    令M2为类别0，M5为类别1

    文件001数量: 47(17/30), 交并: {'CD13', 'CD45', 'CD34', 'CD33', 'CD117'}, {'CD45', 'CD19/CD56', 'CD19', 'CD34', 'CD33', 'CD56/CD19', 'CD16', 'CD15', 'CD19+CD56', 'CD9', 'CD38', 'CD14', 'CD13', 'DR', 'CD7', 'CD56', 'CD19/CD56/CD15', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD64', 'CD117'}
    文件002数量: 35(20/15), 交并: {'CD13', 'CD45', 'CD33'}, {'CD45', 'CD34', 'CD33', 'CD56/CD19', 'CD15', 'CD16', 'CD19+CD56', 'CD9', 'CD38', 'CD14', 'CD36', 'CD13', 'DR', 'CD7', 'CD56', '11b', 'CD19/CD56/CD15', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD10', 'CD64', 'CD117'}
    文件003数量: 24(10/14), 交并: {'CD45'}, {'CD45', 'CD19/CD56', 'CD34', 'CD33', 'CD56/CD19', 'CD22', 'CD15', 'CD19+CD56', 'CD4', 'CD5', 'CD38', 'CD13', 'CD8', 'DR', 'CD7', 'CD56', 'CD3', 'CD19/CD56/CD15', 'CD11B', 'CD20', 'HLA-DR', 'CD117'}
    
    文件004数量: 19(7/12), 交并: {'CD45'}, {'HLA-DR', 'CD45', 'CD34', 'CD33', 'CD15', 'CD56/CD19', 'CD19+CD56', 'CD9', 'CD38', 'CD13', 'CD71', 'DR', 'CD7', 'CD11B', 'CD123', 'CD235', 'CD2', 'CD117'}
    文件005数量: 13(5/8), 交并: {'CD45'}, {'CD45', 'CD34', 'CD33', 'CD15', 'CD56/CD19', 'CD19+CD56', 'cCD79A', 'CD38', 'CD79A', 'CD13', 'cCD79a', 'cCD3', 'DR', 'CD7', 'CD3', 'CD19/CD56/CD15', 'CD11B', 'MPO', 'HLA-DR', 'CD79a', 'CD117'}

    查看第一管数据里面能取多少蛋白，剩下的用0填充试试（因为完全的交集太少了）
    第一管数据中的蛋白并集为：
    {'CD45', 'CD19/CD56', 'CD19', 'CD34', 'CD33', 'CD56/CD19', 'CD16', 'CD15', 'CD19+CD56', 'CD9', 'CD38', 'CD14', 'CD13', 'DR', 'CD7', 'CD56', 'CD19/CD56/CD15', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD64', 'CD117'}
    去掉多种混用的情况后：
    {'CD45', 'CD19', 'CD34', 'CD33', 'CD16', 'CD15', 'CD9', 'CD38', 'CD14', 'CD13', 'DR', 'CD7', 'CD56', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD64', 'CD117'}
    去掉极少使用的蛋白后：'CD16', 'CD15', 'CD9', 'CD14', 'CD123', 'HL-DR', 'CD64', 以及错误的蛋白名称'HL-DR'要纠正为'HLA-DR'，还剩
    {'CD45', 'CD19', 'CD34', 'CD33', 'CD38', 'CD13', 'DR', 'CD7', 'CD56', 'CD11B', 'HLA-DR', 'CD117'}共12个蛋白标签
    再加上物理标签：
    {'FSC-A', 'FSC-H', 'SSC-A', 'CD45', 'CD19', 'CD34', 'CD33', 'CD38', 'CD13', 'DR', 'CD7', 'CD56', 'CD11B', 'HLA-DR', 'CD117'} 
    ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR',       'HL-DR']3物理+12蛋白
    这里不小心空白通道DR忘记去掉了，不过正好看看能不能用来说明分析结果的正确性
    

    第二管数据中的蛋白并集为：
    {'CD45', 'CD11B', 'CD56', 'CD34', 'SSC-A', 'CD123', 'FSC-A', 'CD117', 'CD13', 'CD56/CD19', 'CD36', 'CD15', 'CD64', 'HL-DR', 'FSC-W', 'CD19+CD56', 'CD19/CD56/CD15', 'CD14', 'DR', 'SSC-H', 'CD7', 'CD16', 'FSC-H', 'CD10', 'CD38', 'CD9', 'CD33', 'HLA-DR', '11b', 'SSC-W'}
    去掉多种混用的情况后：
    {'CD45', 'CD11B', 'CD56', 'CD34', 'SSC-A', 'CD123', 'FSC-A', 'CD117', 'CD13', 'CD36', 'CD15', 'CD64', 'HL-DR', 'FSC-W', 'CD14', 'DR', 'SSC-H', 'CD7', 'CD16', 'FSC-H', 'CD10', 'CD38', 'CD9', 'CD33', 'HLA-DR', '11b', 'SSC-W'}
    去掉极少使用的蛋白后：
    {'CD45', 'CD11B', 'CD56', 'CD34', 'SSC-A', 'FSC-A', 'CD117', 'CD13', 'HL-DR', 'SSC-H', 'CD7', 'FSC-H', 'CD38', 'CD33', 'HLA-DR', '11b'}
    排序后：
    {'FSC-A', 'FSC-H', 'SSC-A', 'CD7', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'HLA-DR',       'HL-DR', '11b'}，错误的蛋白名称'HL-DR'要纠正为'HLA-DR'，11b->CD11B
    ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'HLA-DR',       'HL-DR', '11b']，错误的蛋白名称'HL-DR'要纠正为'HLA-DR'，11b->CD11B

### 数据分析
    荧光的最大值为1023，最小值为0，方便用min-max归一化。


### 数据预处理
    min-max归一化
    去除SSC(side scatter)-A为纵坐标的离群点
    去除FSC(forward scatter)-A和FSC-H的离群点
    艾宾浩斯记忆曲线

### 结果记录
#### 第一管数据
    不归一化，用DNN，accuracy后面在78徘徊。归一化后，accuracy来到了95.27。问题：不同病人个体之间泛化性差。把batchnorm换成dropout效果差不多，目前dropout rate从0.1到0.5效果都差不多，原因不明。

    然后打算把网络简化一下，少一点节点试试,dropout 0.5/效果也不太好，试试数据清洗了，由于不同病人细胞的斜率不一样，所以猜测打乱病人样本是不好的选择，但可以尝试。
    有没可能是分类的loss=ylogy 出现了极端值比如0log0导致的训练准确率断崖式下跌

    dropout rate调整为0.4的时候，能够在test set准确率为98.23%，带dropout的train set准确率为80+%。调整一些其他超参，看看稳定性如何，不好。
    事实证明，经过病人打乱后

#### 第二管数据
    混合训练测试(Shuffle)的情况下，准确率为100%

    不shuffle的情况下，


### 意义
    如果一定要讲打乱各个病人的数据以后高准确率的意义，那可能就是当拥有了足够多的病人数据以后，就可以忽略各种样本的不确定性了。以及，仅凭借第一管的信息是可以区分的，只是数据量不够，而且当特征增多，网络的判断更准确，人却更难把握。并且仅通过第一管数据就可以判断出来了。

    问医生 是否有非荧光的散射。
    通道的荧光，是积分的荧光还是哪种荧光，前向的，侧向的。

    减少管数，减少成本，



## 无监督
弄一个最终loss，为聚类细粒度结果的集合，同亚型病人的集合是否有相似的分布（分布相似度）
蛋白联合表达

### 层次聚类-Hierarchical Clustering
https://blog.csdn.net/onroadliuyaqiong/article/details/119053833  它一次性地得到了整个聚类的过程，只要得到了上面那样的聚类树，想要分多少个cluster都可以直接根据树结构来得到结果，改变 cluster数目不需要再次计算数据点的归属

### Clustering using K-Means
https://zhuanlan.zhihu.com/p/619739126?utm_id=0
手肘法：通过画出不同K值与SSE值的折线图，若SSE值下降过程中存在“肘点”（下降速度骤减的拐点处），该点所对应的K值即合适的聚类数。类似DCEC用卷积层替换全连接层后，网络对于文本等数据类型的聚类能力有明显下降，但提高了图像类型（保留了图像的局部特征）。
model.inertia_ 是K-means聚类算法中的一个属性，它代表了各数据样本到其所属簇中心点的距离之和，也被称为误差平方和（Sum of Squared Errors, SSE）。一般而言，在执行K-means算法时，我们希望误差平方和越小越好。

### Clustering using Spectral
Unable to allocate 591. TiB for an array with shape (9016000, 9016000)

### Clustering using Subspace Clustering
由于不稳定性，这种子空间线性表达的方法可能不适用。

### Clustering using GMM

### Clustering using KL

### Clustering using DBSCAN
能够识别任意形状的聚类，并且对噪声点具有鲁棒性。min_samples=2才不会把所有点都认为是噪点。