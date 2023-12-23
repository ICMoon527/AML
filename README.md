# AML

## 数据处理
### 数据获取
    拿到 ['CD19+CD56 FITC-A', 'CD13 PE-A','CD117 PerCP-Cy5-5-A', 'CD33 PE-Cy7-A', 'CD34 APC-A', 'CD7 APC-R700-A',
    'CD38 APC-Cy7-A', 'DR V450-A', 'CD45 V500-C-A', 'CD11B BV605-A'] 这些蛋白的数据，包含17个M2和20个M5病人，令M2为类别0，M5为类别1

### 数据分析
    荧光的最大值为1023，最小值为0，方便用min-max归一化。


### 数据预处理
    min-max归一化
    艾宾浩斯记忆曲线