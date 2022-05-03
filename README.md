# TSH
Terminal-Satellite Handover

# USING

contents:
- STK场景创建
- STK数据导出
- python数据预处理
- 单次仿真
- 批量仿真
- bash 批量
- 测量


### 1.场景创建



这里有两个access, 一个是链路预算需要的(参考[链路预算视频](https://www.bilibili.com/video/BV1Ru41127zM/)
), 即`place::motor::receiver --> satellite`;
另一个是吻合星地链路的视线范围(Line of Sight)的access, 即 `place::motor --> satellite`.
![motor_prop.png](./fig/motor_prop.png)
result:
![](./fig/angle.png)

如果使用这种方法stk默认在卫星离境时候自动选择剩余服务时间最长的(MST)卫星, 为了得到终端对多个卫星的邻接数据, 还是将地面终端设置一个张角为卫星仰角补角的sensor, 即可实现.

### 2. 数据导出

需要先通过stk生成access数据, 此处有一个较小的例子, 单颗卫星对地面站的access. 这里首先创建场景, 如下



``` python
# chengdu 		 	#place
# 	motor1		 	#sensor
#		reciver1 	#reciver


#sat				#satellite
#	motor2		 	#sensor
#		transmitter	#transmitter

```
文章中称为 coverage intervals table.

### 3.数据预处理

预处理后, 信息量不变,  得到处理后的数据`*.csv`格式

### 4. 施加算法`main.py`
根据设置, 随机生成子场景的时刻和持续时间, 载入内存, 将这部分数据, 施加算法, 得到一个procedure.

- config file: config.yaml


- 生成逻辑: (city, sim_duration, random_seed, algorithm) ==> procedure

- 生成文件
    ```bash
    yymmdd-hhmmss/
        |- final_value.csv
        |- stat_results.json
        |- settings.yaml
        |- solution.png
     
    ```

### 5. 批量处理 `batch_run.py`

根据随机种子列表, 对`step.4`反复, 得到一个instance, 保存到文件夹.
- 生成逻辑: (city, sim_duration, **random_seed, algorithm**) ==> instance
- config file: batch_run_config.yaml, [city]_[simduration].yaml
- control flow:
    ``` python
    for seed in seeds:
        for alg in algorithm
            do
    ```
- 生成文件
    ```bash
    yymmdd-hhmmss/
        |- seed_alg.csv
        |- seed_alg.json
        |- settings.yaml
        |- ...
     
    ```
### 6. 终端再批量`bash.sh`

对 step.5 的脚本, 运行多次, 得到不同的instance
- config file: None
- 生成逻辑: (**city, sim_duration, random_seed, algorithm**) ==> instances
- control flow:
  ```shell
    bash batch_config.yaml city-sim_duration #dir
  ```
- 生成文件
    ```bash
    yymmdd-hhmmss/
    yymmdd-hhmmss/
    yymmdd-hhmmss/
    ...
    ```
### 7. measurement
连接性测量, QoS测量

- config file: measurement.yaml
- 掉话率测量伪代码, 输入为instance(s) from step.6
- control flow:
    ```python

        for batch_calls, batch_procedures in zip([calls_container, procedures_container]):
            for procedure in batch_procedures:
                procedure.inject(batch_calls)
    ```

掉话率测量过程如图所示

![](./fig/pdrop_pip.png)

掉话率测量结果如下

![](./fig/pdrop_results.png)

# 源码解释
`TODO`