# Reinforcement Learning  

## 基本名词

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