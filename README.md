# Indoor-positioning 室内定位

室内定位算法：T-WKNN方法源码

For my indoor positioning algorithm : T-WKNN.

**2018.11.12更新**

1.实验将按照月份顺序进行分类，而非原来的按照测试集的序号进行分类。

2.引入r参数，目的在于提高边缘数据的分类准确率 

**2019.4.22更新**

1.VOTEP 改为 T-WKNN 算法源代码上传 

增加内容：

① AP selection

② 理论证明 tolerance-WKNN

③ 对比方法增加 WKNN M-WKNN GK LS-SVM

2.补充部分注释

**2019.7.5更新**

论文被录用XD

1.使用自适应曼哈顿距离ACMD来匹配指纹

2.AP selection 针对每个RP进行，形成局部最优AP选择集合

3.实验结果多方面展示 CDF图 均值误差 均值方差 均方根误差等

![image](https://github.com/XBB1995/Indoor-positioning/raw/XBB1995-2019-4-22/image/result.jpg)

