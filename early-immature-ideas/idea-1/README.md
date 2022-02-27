# IDEA-1

> 这是最直接的想法：
> 
> 找一个现成的比较简洁的强化学习框架，看看它是怎么执行的，并尝试用它验证自己的想法。


* ElegantRL - 强化学习算法
* gym - 环境模拟
* rqdata - 米筐金融数据API(数据源)


在查找强化学习框架的时候看到了哥大开源的[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
它默认使用GYM作为交互环境。

尝试自行扩展一个GYM环境——TinyMarket（如果为了开发完善，很容易把注意力聚焦在实现细节。这偏离了此次研究本意）。
后期发现已经有现成的 [gym-anytrading](https://github.com/AminHP/gym-anytrading) ,
主体思路与TinyMarket非常类似，对比发现实现思路也类似，只是更为简洁，简化的思路更适合作为研究使用。
自己实现的考虑了太多现实细节，增加了部分不必要的复杂度。

---
## 做过的尝试
### 动作空间
0-观望
1-看多
2-看空

仓位管理方式： 
- 看多（1） - 空仓直接开一手多。当前持有多头仓位则加一手多；持有空头，平掉最早开的一张空单。
- 看空（2） - 空仓直接开一手空。当前持有空头仓位则加一手空；持有多头，平掉最早开的一张多单。

可以保证任何时刻当前持仓只有一个方向，只是仓位不同。

### 状态空间
包括三部分： 市场状态、仓位状态 和 相关状态
##### - 市场状态
就是市场的当前tick数据经过转换可以得到的状态。为了提高效率这里可以对他们进行预处理。
##### - 仓位状态
市场信息不随着动作而变化，变化的就是持仓状态。当前持仓的信息，包括头寸方向，头寸大小，保证金，收益等等。
##### - 其他相关状态
比如在计算奖励的时候考虑了某些指标，那么就把这个指标也添加到状态信息里作为返回（奢望Agent能学到奖励的细节？？）。
比如设计的几种奖励扣减方式，会把扣减的金额或者是扣减的原因等作为相关状态加入到状态信息中返回。

---

## 市场状态的几种情况
+ 原始数据
+ 简单加工
+ 简化数据
+ 考虑特征缩放

### 原始数据
数据取自rqdata金融API

2010年的豆粕09合约示例数据（data_sample/M1009.h5)

```
date              20100602.0  # 日期 2010-06-02
time              90027120.0  # 时间 09：00：27.120
last              27470000.0  # 最新价 * 10000
high              27490000.0  # 当日最高价 * 10000
low               27460000.0  # 当日最低价 * 10000
volume                 452.0  # 累计成交量
open_interest       525114.0  # 持仓量
total_turnover    12420060.0  # 累计成交额
a1                27470000.0  # 卖一价格
b1                27460000.0  # 买一价格
a2                       0.0
b2                       0.0
a3                       0.0
b3                       0.0
a4                       0.0
b4                       0.0
a5                       0.0
b5                       0.0
a1_v                    48.0  # 卖一量
b1_v                    13.0  # 买一量
a2_v                     0.0
b2_v                     0.0
a3_v                     0.0
b3_v                     0.0
a4_v                     0.0
b4_v                     0.0
a5_v                     0.0
b5_v                     0.0
```

### 简单加工
去掉无用数据（卖卖二三四五）

把时间数据换成时间进度数据

价格数据换成正常价格

最高价最低价信息变成当前价格在最高最低价差值的什么位置

累积数据换成每个时刻的变化量数据

成交量、价格取1分钟3分钟5分钟的均值和标准差

```
date                  2.010012e+07  # 日期 2010-01-18（只是用了科学计数法表示了数字）
time                  9.083015e+07  # 时间 09：08：30.147
time_progress         2.557200e+00  # 时间进度，当天最后一个时间为100，第一个时刻为0，中间时刻为0-100之间的某个值
last_price            2.902000e+03  # 最新价
price_scale           7.000000e-01  # 当前价格在当前最高价与最低价之间的位置 （当前 - 最低）/（最高 - 最低）
price_mean_1m         2.901758e+03  # 一分钟价格均值
price_std_1m          5.500000e-01  # 一分钟价格标准差
price_mean_3m         2.902356e+03  
price_std_3m          9.057000e-01
price_mean_5m         2.902635e+03
price_std_5m          1.639000e+00
ask                   2.903000e+03  # 最新卖价
bid                   2.902000e+03  # 最新买价
ask_gap               1.000000e+00  # 最新卖价与最新价差值
bid_gap               0.000000e+00  # 最新买价与最新价差值
ask_volume            9.270000e+02  # 卖量
bid_volume            6.190000e+02  # 买量
voi                  -4.300000e+01  # 根据Shen（2015）构建的交易量订单流不平衡（Volume Order Imbalance）指标 
weibi                -1.992238e-01  # 委比 = [委买数－委卖数]/[委买数＋委卖数]×100％
open_interest_diff   -1.200000e+01  # 每个时刻的持仓量变化
turnover              1.100000e+02  # 每个时刻的交易量（注意已经不是累计值了）
turnover_mean_1m      3.801792e+02
turnover_std_1m       6.682346e+02
turnover_mean_3m      4.203681e+02
turnover_std_3m       6.263943e+02
turnover_mean_5m      5.310892e+02
turnover_std_5m       8.285374e+02
```

每个状态数据除了前两个表示当前时间的数字，其他都会作为观察状态被返回

经过训练发现到最后就会躺平，开始怀疑是状态数据取得不对，所以尝试把数据取值换个方式

引入指标和变化率

```
date                      2.021112e+07
time                      9.082035e+07
last_price                3.186000e+03
ask                       3.187000e+03
bid                       3.186000e+03
ask_volume                1.560000e+02
bid_volume                3.730000e+02
ask_volume_roc            1.923077e-02
bid_volume_roc            1.608579e-02
tsf                       3.184586e+03   # TSF指标
turnover                  1.000000e+01
turnover_roc              5.000000e-01
open_interest_diff       -1.000000e+00
open_interest_diff_roc    1.000000e+00
macd                     -7.143064e-01  # MACD指标
macd_roc                 -3.418672e-03
macd_signal              -6.755254e-01
macd_signal_roc           1.290084e-03
macd_hist                -3.878105e-02
macd_hist_roc            -8.544027e-02
rsi                       4.994675e+01  # rsi指标
rsi_roc                  -7.973927e-03
obv                       9.383500e+03  # OBV指标
obv_roc                  -1.065700e-03
k                         2.525464e+01  # KDJ指标
k_roc                    -1.143103e-02
d                         2.853443e+01
d_roc                    -4.691501e-03
j                         1.869504e+01
j_roc                    -3.200425e-02
```
意义不大，考取使用LSTM试试

### 简化数据

加入了rsi指标

去掉了时间进度这种信息（每条数据不带时间信息了）

因为考虑使用LSTM所以不带太多均值信息了

```
date               2.021112e+07
time               9.082035e+07
last_price         3.186000e+03
ask                3.187000e+03
bid                3.186000e+03
ask_volume         1.560000e+02
bid_volume         3.730000e+02
ask_gap            1.000000e+00
bid_gap            0.000000e+00
rsi_2m            -2.736977e-01
rsi_5m             8.636927e-01
voi                3.000000e+00
weibi              4.102079e-01
open_interest_d   -1.101666e-02
turnover_v_5m      1.597414e+00
```

### 特征缩放

看看是不是因为数据取值范围差异太大导致训练不出结果

把所有数据尽量缩放到-10到10之间

通过装饰器类GymEnvFeatureScaling完成

```
# 缩放量价信息数据到10以内
transaction_state['last_price'] = transaction_state['last_price'] / 1000
transaction_state['ask'] = transaction_state['ask'] / 1000
transaction_state['bid'] = transaction_state['bid'] / 1000
transaction_state['ask_volume'] = transaction_state['ask_volume'] / 100
transaction_state['bid_volume'] = transaction_state['bid_volume'] / 100
```
---

## 持仓状态

- 较详细信息
- 简化信息

### 比较详细的信息
主要是和持仓相关的所有信息

基本类GymEnvBase中_check_position_info方法实现

```
return {
        'position': position,
        'floating_pl': floating_pl,
        'closed_pl': self.closed_pl,
        'amount': amount,
        'risk': risk,
        'free_margin': free_margin,
        'margin': margin
    }
```
后来感觉返回太多无效信息，而且数据范围很大 所以进行了简化

### 简化信息

只保留比较重要的 头寸方向信息 风险率信息 持仓盈亏 三个数据

装饰类GymEnvFeatureScaling中复写的_observation方法实现

``` 
拼接数据并返回结果
return np.array(tuple(transaction_state)[2:]
                + (
                    position_state['position'],
                    float(position_state['risk']),
                    float(position_state['floating_pl'] / 10))
                )
```

### 相关信息

对数据奖励行为的管理使用装饰器类完成，不同的装饰器类奖励会将对奖励有影响的数据作为状态的相关数据附加到状态数据后面

- **timeout_risk** - 订单超过最大持仓时间的风险程度 当前持仓秒数/最大持仓秒数
- **adequacy** - 保证金充足程度 可用保证金/当前每手持仓的保证金    （因为没有后来想想也可以直接用“保证金率”替代）
- **action_imbalance** - 动作不平衡程度，比如开很多多单不开空单 (因为希望可以自己平仓而不是等超过最大持仓时常自动平仓)
- **fine** - 观望罚金（不希望Agent一直观望而不做动作，所以从开始到第一个动作之前会随着时间增加给出惩罚

---

# 类

## 基本类

- **GymEnvBase** - 基类 最基本功能
  + 选定开始时间后逐个执行交易并返回状态信息
  + 维护持仓状态： 超过最大持仓时间（5分钟）自动平仓
  + 维护持仓状态： 距离收盘一定时间内（6分钟）只允许平仓不允许开仓
  + 维护持仓状态： 距离收盘一定时间（1分钟）清仓
  + Reward直接是权益的损益
- **GymEnvDaily** - 按照日期逐笔执行（继承基类） 主要用于回测
  + 日期按照指定范围逐日执行
  + 日内按照时间顺序逐笔执行
  + 收盘或剩余权益无法开仓后推出
- **GymEnvRandom** - 随机进入点执行（继承基类） 主要用于训练
  + 指定范围内选取随机日期
  + 日内随机选一个时间开始
  + 进行操作后恢复空仓时退出（模拟一次 ” 观望 -> 建仓 -> 平仓 “ 的过程）

## 装饰类

- **GymEnvWaitAndSeeWillResultInFines** - 持仓观望会遭到惩罚
  + 用在训练时（但是回测时因为这个会改变返回的观察状态维度所以不能去掉
  + 罚金随着观望时间增长不断增加，开仓后不再增加
  + Reward的基础上扣减一个罚金
  + 会将罚金作为一个相关状态**fine**添加到环境观察状态中返回给Agent
- **GymEnvLongAndShortImbalanceWillResultInFines** - 看多看空动作不平衡会被惩罚
  + 用在训练时（但是回测时因为这个会改变返回的观察状态维度所以不能去掉
  + 罚金随着多空操作数不一致折算一个惩罚
  + 只要不一致每个tick都会进行惩罚，所以不平衡时间越久不平衡程度越大收到损失越大
  + Reward的基础上扣减一个罚金
  + 会将罚金作为一个相关状态**action_imbalance**添加到环境观察状态中返回给Agent
- **GymEnvDoneIfOrderTimeout** - 有订单超时被自动平仓导致本轮探索结束
  + 用在训练时（但是回测时因为这个会改变返回的观察状态维度所以不能去掉
  + 结束会给个较大惩罚作为负奖励
  + 会将超时风险度作为一个相关状态**timeout_risk**添加到环境观察状态中返回给Agent
- **GymEnvDoneIfUndermargined** - 有保证金不足还强制开仓时探索结束
  + 用在训练时（但是回测时因为这个会改变返回的观察状态维度所以不能去掉
  + 结束会给个较大惩罚作为负奖励
  + 会将保证金充足度作为一个相关状态**adequacy**添加到环境观察状态中返回给Agent
- **GymEnvFeatureScaling** - 缩放价格相关参数并改变带回仓位状态数据
  + 训练和测试都需要
- **GymEnvUseSharpRatioAsReward** - 用日夏普率作为奖励
  + 从每tick的资金变化反推日夏普率作为奖励返回
  + 感觉算错了?突变略大？

---

# 结论

## 尝试过三种组合情况

```
# step08
class GymEnvRandomFineWatching(GymEnvWaitAndSeeWillResultInFines,GymEnvRandom)

# step14
class GymEnvRandomIndData(GymEnvLongAndShortImbalanceWillResultInFines,
                          GymEnvWaitAndSeeWillResultInFines,
                          GymEnvFeatureScaling, GymEnvRandom)

# step19
class GymEnvRandomWithSharpeReward(GymEnvDoneIfUndermargined,
                                   GymEnvDoneIfOrderTimeout,
                                   GymEnvLongAndShortImbalanceWillResultInFines,
                                   GymEnvWaitAndSeeWillResultInFines,
                                   GymEnvFeatureScaling,
                                   GymEnvRandom)
```
## 尝试过一种定制用变化率的情况

```
# setp12
GymEnvRandomRocData
```

观察状态中仓位状态简化为 头寸和风险度两个值

## 尝试过LSTM方案

用的是 step14 的环境

---

# 最后结论就是：
# 刚开始会挣扎，但终会躺平
# 状态信息变化，确实可以影响收敛速度，但不影响结论

其实训练过程可以发现，损失稍微缩减后几乎就是随机游走状态不再收敛 
{
模型表现力不够？
或者数据没有指向性，拟合了噪音？
也可能是因为每次都只返回一条状态导致的数据带的历史信息（趋势信息）不足
}

回测数据几乎一直在赔钱
{
因为成交价格用的订单簿价格（卖一价买，买一价卖）而不是当前价格，这样只要交易就会有损失（摩擦过大？）。
此外还考虑了手续费。
}

训练过程参考 checkpoint_TinyMarketGymEnvRandom.zip

训练过程Agent行为大概分为几个阶段：（很像一个标准的韭菜
1. 兴致勃勃 （不停尝试）
2. 收敛  （交易次数减少）
3. 躺平  （不交易了，偶尔诈尸交易下然后继续躺平）

**猜想原因：**

一方面可能是发现躺平收益最稳（后来改成躺平会有较高惩罚也没拦住躺平）


另一方面原因可能因为交易中其实绝大多数时间都是在观望状态（等待开仓机会 或者持仓等待价格变动），所以导致“观望”这个动作的权重会一直上升，而多空操作会因为不同盈利忽上忽下
（看到 gym-anytrading 动作空间只有 多和空两个（不包含观望），可能也是考虑到这个因素？那我这个躺平就不奇怪了 ：）


## 反思整体方案对强化学习理解太浅，过度关注工程性（各种重构就为了类继承JAVA上头了？）而非本质（强化学习到底怎么用在期货高频交易上），对相关工具也没有太深了解。只能算是一次上手实验。

## 论文已延期，希望后续能做出理想结论……
