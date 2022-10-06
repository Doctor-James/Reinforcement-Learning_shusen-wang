# Deep Reinforcement Learning  

[深度强化学习基础](https://www.bilibili.com/video/BV1rv41167yx?p=5&vd_source=8a196b748d509b7735169a013d4b46d8)

## Terminologies  

### State and Action

![image-20221004190209676](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004190209676.png)

**state**是场景

**agent**为智能体，也就是执行操作的对象，此处是马里奥，也可以是机器人，车等

**action**为agent做出的动作

### Policy

![image-20221004190331028](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004190331028.png)

Policy函数，即策略函数，通常是一个概率分布函数，如该ppt中，在当前state下，马里奥选择往左走的策略的概率为0.2。该state下马里奥有三种策略，会在其中随机抽样，随机性使得policy更加灵活，难以被预测

### Reward

![image-20221004190659689](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004190659689.png)

reward为奖励函数，如此处吃到金币+1分，通关+1w分，碰到敌人（game over）扣1w分

### State Transition

![image-20221004190847344](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004190847344.png)

状态转移，顾名思义是old state变为new state，上图P为条件概率密度函数，意思是如果观测到当前的状态S以及动作A，下一个状态变成S一撇的概率

由于env以及action的随机性，所以状态转移具有随机性

### Two Sources of Randomness  

整个过程中的随机性主要来源于两点：

1. 是action的随机性，这个很好理解
2. 是state的随机性，在马里奥的例子中可以理解为敌人移动也是随机的，无法被agent知晓

![image-20221004191123216](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004191123216.png)

![image-20221004191130398](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004191130398.png)

### Trajectory  

整个过程可以认为是

1. 观察state
2. 做出action
3. 观察到新的state并得到奖励（或惩罚）

![image-20221004191345268](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004191345268.png)

### Rewards and Returns  

**returns**定义为未来所有cumulative future reward未来的累积奖励

![image-20221004191725405](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004191725405.png)

但其实之后的奖励对当前时刻不是同等重要的，如选择现在给你100元和一年后给你100元，大部分人会选择立刻得到一百

所以引入折扣回报Discounted return

![image-20221004191918628](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004191918628.png)

gamma介于0到1，为超参数

R 与 S 和 A有关

![image-20221004192502906](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004192502906.png)

所以U 与未来所有S A 有关

![image-20221004192525560](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004192525560.png)

### Value Function

价值函数

**Action-value function**

Ut其实是一个随机变量，他依赖于之后的所有动作和状态，t时刻我们并不知道Ut是什么，所以我们可以对Ut求期望得到一个函数，即Action-value function，动作价值函数，Qpi

![image-20221004193349980](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004193349980.png)

此时我们Ut未知，St和At是变量，且他们的概率分布已知（S-t，A-t分布），则可以用积分的方式把将来的SA对当前时刻Ut的影响通过积分求期望的方式获得

将t时刻之后的随机变量A， S都用积分积掉，之后得到的Qpi就只与当前时刻t的SA以及pi有关  

Action-value function的实际意义，一直policy函数pi以及当前t时刻的s，则可以通过Action-value function Qpi对每个action打分，看做出哪个action，最终Ut的期望最高



**Optimal Action-value function**

最优动作价值函数

之前说的Action-value function与pi，SA有关，而我们有很多个policy函数pi，我们要使得Action-value function最大，则可以做一个动态规划，取得最优的policy函数pi，使得Action-value function最大，这样就可以消除pi对Action-value function的影响（因为policy函数pi已经确定了，不再是变量），得到一个最优的Action-value function，即Optimal Action-value function

![image-20221004194243499](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004194243499.png)



**State-value function**

![image-20221004195722930](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004195722930.png)

![image-20221004195849125](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004195849125.png)

将Qpi对A求期望（将A当作随机变量），从而消掉A

物理意义在于可以评估目前状态的胜算

**Conclusion**

![image-20221004200015739](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004200015739.png)

### How does AI control the agent

1. policy-based learning 策略学习：已知pi，S，可以求得每个A的概率，再随机抽样
2. value-based learning 价值学习：求Optimal Action-value function Q-star

![image-20221004200908262](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221004200908262.png)

## Value-Based Reinforcement Learing

### Deep Q-Network (DQN)  

回顾上节课讲的**Optimal Action-value function** Q_star，Q_star的作用是判断在当前state下，哪个action带来的未来reward总和的期望越大。而Q_star往往是不能直接得到的，价值学习的基本想法就是通过学习一个函数（神经网络）来近似Q_star

![image-20221005194452241](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005194452241.png)

![image-20221005194616135](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005194616135.png)

### Temporal Difference (TD) Learning  

王老师举了一个预测开车时间的例子

![image-20221005195046093](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005195046093.png)

如图用梯度下降法，比较naive，需要完成一次旅程才能update model

![image-20221005200008765](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005200008765.png)

使用TD learing，走到路程中间（300min处），再使用模型预测一次，预测值为600min，容易想到离终点越接近，该估计会越准，所以可以认为300+600=900的估计比一开始的1000更准，1000与900的差称为TD error



![image-20221005201007771](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005201007771.png)

TD learning可以运用的场景，即可以写作estimate = estimate+actual的形式，其中的等于是我们最理想的情况，即TD error等于0

回顾discount return

![image-20221005201235609](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005201235609.png)

![image-20221005201520732](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005201520732.png)

![image-20221005201628201](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005201628201.png)

## Policy-Based Reinforcement Learing

同样，策略函数pi也是难以直接获得的，所以需要通过神经网络来近似，此神经网络被称为policy network

![image-20221005202053448](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202053448.png)

![image-20221005202208219](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202208219.png)

回顾之前学的state-value fuction Vpi，Vpi是Qpi对action求期望，可以表示在当前state下的胜算

![image-20221005202502198](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202502198.png)

Vpi可以写作下图形式

![image-20221005202555409](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202555409.png)

V（s，theta）可以度量状态S和策略网络theta的好坏，给定状态s，策略网络theta越好，则V越大

所以我们可以把目标函数设置为

![image-20221005202802328](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202802328.png)

策略网络theta越好，J_theta越大

![image-20221005202923690](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005202923690.png)

此处使用的是梯度上升算法，因为我们是想要目标函数越大越好（对比loss函数）

### Policy gradient

此处Policy gradient的求导会用到高数和概率论，不太好记录，建议多看看ppt和视频

![image-20221005203522364](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005203522364.png)

![image-20221005203539067](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005203539067.png)

我们得到了以上两种方式来表示policy gradient

对于动作是离散形式，可以使用Form1枚举计算

![image-20221005203905361](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005203905361.png)

对于action是连续形式，则需要积分，但Qpi是一个神经网络非常复杂，不能直接积分得到解析解，所以需要使用蒙特卡洛算法近似（此处需要补概率论...）

具体可见[蒙特卡洛近似 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/36033544)

[蒙特卡洛近似的一些例子](https://blog.csdn.net/qq_38689352/article/details/119489072)

[数学理论—— 蒙特卡洛近似](https://blog.csdn.net/Cyrus_May/article/details/123967769?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-123967769-blog-119489072.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-123967769-blog-119489072.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=1)

![image-20221005204835348](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005204835348.png)

1. 随机抽样一个a_hat，抽样是根据概率密度函数pi抽的
2. 计算g（a_hat,theta）

![image-20221005205057961](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005205057961.png)

可以知道g（A，theta）对A求期望即为V的导数

且g（a_hat,theta）是V求导的无偏估计（？）

则可以用g（a_hat,theta）来近似V求导

蒙特卡洛算法就是通过抽取一个或多个样本对期望进行近似



整个流程如下图所示

![image-20221005205700825](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005205700825.png)

但还有一个问题没有解决，即由于Qpi无法得知，所以qt不能直接得到，有如下两种方法来近似

1. Reinforce算法

因为Qpi的Ut的期望，所以可以用ut来近似Qpai，即近似qt，该方法需要完整玩完一局游戏才能对策略函数进行更新

![image-20221005210215805](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221005210215805.png)

2. actor-critic method  

用神经网络近似qt，下节课具体讲解



## Actor-Critic Methods  

![image-20221006161622328](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006161622328.png)

之前学习的Policy network，策略网络，使用 net去近似policy函数pi，他控制agent的运动action，被称为actor

Value network不直接控制agent，而是对动作打分，被成为critic

![image-20221006162228152](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006162228152.png)

![image-20221006162251382](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006162251382.png)

更新两个network的目标是不同的

1. 更新policy network的目标是增大V的值（因为V是对未来return的期望）
2. 更新value network的目标是使得q的打分更准

![image-20221006162454234](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006162454234.png)

![image-20221006163000822](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006163000822.png)

### Summary

![image-20221006163930669](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006163930669.png)

## AlphaGO

![image-20221006164734902](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006164734902.png)

![image-20221006164742895](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20221006164742895.png)

**Imitation learning**,模仿学习  
