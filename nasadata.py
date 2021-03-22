# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:08:14 2020

@author: ckl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat  # 用于加载mat文件
from scipy import integrate   # 用于计算积分
#from sko.PSO import PSO       # pso算法


'''数据预处理'''
# 电池额定容量为2Ah
RC = 2


# 读取NASA锂电池试验数据文件放入dir_list
NCA_data = os.listdir('H:\\BatteryDataSet\\BatteryDataSet')
dir_list = []  # 文件夹列表
for i in range(len(NCA_data)):                  # 生成路径 dir——list
    dir_list.append('H:\\BatteryDataSet\\BatteryDataSet' + '\\' + NCA_data[i])


# 原始电池数据提取函数
def getBattery(name):
    '''读取电池mat格式数据'''
    data =  loadmat(name)
    key_name = list(data.keys())[-1]
    mat_data = data[key_name]   
    return mat_data
 

# 格式转换函数
def to_df(mat_data):
    """将mat数据格式转换为dataframe"""
    
    # Features common for every cycle
    cycles_cols = ['type', 'ambient_temperature', 'time']

    # Features monitored during the cycle
    features_cols = {
        'charge': ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                'Current_charge', 'Voltage_charge', 'Time'],
        'discharge': ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                    'Current_charge', 'Voltage_charge', 'Time', 'Capacity'],
        'impedance': ['Sense_current', 'Battery_current', 'Current_ratio',
                    'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct']
    }

    # Define one pd.DataFrame per cycle type
    df = {key: pd.DataFrame() for key in features_cols.keys()}

    # Get every cycle
    print(f'Number of cycles: {mat_data[0][0][0].shape[1]}')
    cycles = [[row.flat[0] for row in line] for line in mat_data[0][0][0][0]]

    # Get measures for every cycle
    for cycle_id, cycle_data in enumerate(cycles):
        tmp = pd.DataFrame()

        # Data series for every cycle
        features_x_cycle = cycle_data[-1]

        # Get features for the specific cycle type
        features = features_cols[cycle_data[0]]
        
        for feature, data in zip(features, features_x_cycle):
            if len(data[0]) > 1:
                # Correct number of records
                tmp[feature] = data[0]
            else:
                # Single value, so assign it to all rows
                tmp[feature] = data[0][0]
        
        # Add columns common to the cycle measurements
        tmp['id_cycle'] = cycle_id
        for k, col in enumerate(cycles_cols):
            tmp[col] = cycle_data[k]
        
        # Append cycle data to the right pd.DataFrame
        cycle_type = cycle_data[0]
        df[cycle_type] = df[cycle_type].append(tmp, ignore_index=True)
    
    return df



'''对放电数据进行操作'''
# 放电数据修正 
def discharge_corr(nobattery):
    '''对放电数据进行操作'''
    # 放电数据中有容量数据，用其转换为SOH
    discharge = nobattery['discharge']
    discharge['SOH'] = discharge['Capacity'] / RC * 100
    
    # 筛掉容量为0的行
    discharge = discharge.drop(discharge[discharge['Capacity']==0].index)    
    
    # 按循环充放电周期修正索引
    discharge_cycle = discharge.drop_duplicates(subset=['id_cycle'],keep='last')
    discharge_cycle = discharge_cycle.reset_index(drop = True)
    discharge_cycle = discharge_cycle.reset_index() 
  
    return discharge_cycle



'''对充电数据进行操作'''
# 充电片段提取函数
def charge_trip(df): # 传入电池充电数据
    '''对充电数据按循环次数找到每次充电的开头索引'''
    df_trip_start = df.loc[(df['id_cycle'] != df.shift(1)['id_cycle'])]
    start_index_list = df_trip_start.index.to_list() # 区间首索引
    if start_index_list[0] != 0 : start_index_list.insert(0,0)
    if start_index_list[-1] != df.index[-1] : start_index_list.append(df.index[-1] +1) 
    for i in range(0,len(start_index_list)-1):
        df.loc[start_index_list[i] :start_index_list[i+1], 'trip'] = i+1
    df['trip'] = df['trip'].astype(int)
    return df, start_index_list


# 恒流充电筛选函数
def get_charge_trip(charge,charge_trip_index):
    '''得到每段恒流充电片段的时间、电流、电压'''
    charge_trip = []
    charge_df = pd.DataFrame(columns = ['trip', 'time', 'volt', 'current'])
    for i in range(len(charge_trip_index)):
        try: 
            trip = charge.loc[charge_trip_index[i]:charge_trip_index[i+1]-1, 'trip']
            time = charge.loc[charge_trip_index[i]:charge_trip_index[i+1]-1, 'Time']
            volt = charge.loc[charge_trip_index[i]:charge_trip_index[i+1]-1, 'Voltage_measured']
            current = charge.loc[charge_trip_index[i]:charge_trip_index[i+1]-1, 'Current_measured']
            charge_df = pd.concat([trip, time, volt, current],axis=1)
            # 取恒流充电部分的数据
            charge_df1 = charge_df[(charge_df['Current_measured'] < 1.52) & (charge_df['Current_measured'] > 1.49)]
            charge_trip.append(charge_df1)          
        except:
            break    
    return charge_trip


# 计算电压积分函数
def calc_VoltIntegral(charge_trip):
    '''计算电压积分'''
    VoltIntegral = []
    for i in range(len(charge_trip)):
        x = charge_trip[i]['Time']
        y = charge_trip[i]['Voltage_measured']
        result = integrate.trapz(y, x)
        if result > 0:
            VoltIntegral.append(result)
        # 筛掉第一个和最后一个片段的电压积分
        vi = pd.DataFrame(VoltIntegral,columns=['vi'])[1:-1]
    return vi


# 计算电压积分函数
def calc_VoltIntegral0(charge_trip):
    '''计算3.7v-4.2v间的电压积分'''
    VoltIntegral = []
    for i in range(len(charge_trip)):
        charge_trip0 = charge_trip[i][(charge_trip[i]['Voltage_measured'] < 4.2) & (charge_trip[i]['Voltage_measured'] > 3.7)]
        x = charge_trip0['Time']
        y = charge_trip0['Voltage_measured']
        result = integrate.trapz(y, x)
        if result > 0:
            VoltIntegral.append(result)
        # 筛掉第一个和最后一个片段的电压积分
        vi = pd.DataFrame(VoltIntegral,columns=['vi'])[1:-1]
    return vi


# 电压积分与soh与循环次数的关系
def soh_vi_cycle(vi,discharge):
    df = pd.DataFrame(columns = ['cycle', 'soh', 'vi'])
    vi = pd.DataFrame(vi,columns=['vi'])
    cycle = discharge['index'][1:-1]
    soh = discharge['SOH'][1:-1]
    df = pd.concat([cycle,soh,vi],axis=1)
    return df


# 电池第二次充放电循环数据 用于展示电池充放电过程
def charging_process(data_df):
    '''电池第二次充放电循环数据'''
    charge = data_df['charge'].loc[data_df['charge']['id_cycle']==2,:]
    discharge = data_df['discharge'].loc[data_df['discharge']['id_cycle']==3,:]
    discharge['Time'] = discharge['Time'] + charge['Time'].iloc[-1]
    cycle = pd.concat([charge, discharge])    
    return cycle



# 遍历出所有电池的数据，并对放电电流进行修正
all_data_df = []                        # 所有电池数据
all_discharge = []                      # 所有电池放电数据
all_charge = []                         # 所有电池充电数据
all_charge_trip_index = []              # 所有充电数据每段trip开头索引
all_charge_trip = []                    # 所有电池的恒流充电阶段数据
for i in range(len(dir_list)):
    # 转换所有电池数据格式 mat -- df
    data_mat = getBattery(dir_list[i]) 
    data_df = to_df(data_mat)
    # 充电数据操作
    charge = charge_trip(data_df['charge'])[0]
    charge_trip_index = charge_trip(data_df['charge'])[1]
    charge_trip_data = get_charge_trip(charge,charge_trip_index)
    # 放电数据操作
    discharge = discharge_corr(data_df)
    # 放入列表
    all_data_df.append(data_df)
    all_charge.append(charge)
    all_charge_trip_index.append(charge_trip_index)
    all_charge_trip.append(charge_trip_data)
    all_discharge.append(discharge)
    

# 计算电压积分
all_vi = []                             # 所有电池充电片段的电压积分
for i in range(len(all_charge_trip)):
    # 筛掉第一个和最后一个片段的电压积分
    vi = calc_VoltIntegral(all_charge_trip[i])
    all_vi.append(vi)


# 计算3.7v-4.2v间的电压积分
all_vi0 = []
for i in range(len(all_charge_trip)):
    # 筛掉第一个和最后一个片段的电压积分
    vi = calc_VoltIntegral0(all_charge_trip[i])
    all_vi0.append(vi)    


# 电压积分和soh和循环次数关系
all_soh_vi_cycle = []                   
for i in range(len(all_vi)):
    soh_vi_cycle0 = soh_vi_cycle(all_vi[i],all_discharge[i])
    all_soh_vi_cycle.append(soh_vi_cycle0)



'''作图'''
# 作图1：电压、电流-时间  #以五号电池的第二次充放电循环数据作为展示
cycle = charging_process(all_data_df[0]) # 电池第二次充放电循环数据
fig1, ax1 = plt.subplots() 
ax2 = ax1.twinx()
x1 = cycle['Time'] 
y1 = cycle['Voltage_measured']
y2 = cycle['Current_measured']
ax1.set_xlabel("时间（s）",fontproperties = 'KaiTi',fontsize = 15)
ax1.set_ylabel("电压（V）",color='g',fontproperties = 'KaiTi',fontsize = 15)
ax2.set_ylabel("电流（A）",color='b',fontproperties = 'KaiTi',fontsize = 15)
plt.xlim([1, 13800])
plt.ylim([-3, 2])
l1, = ax1.plot(x1,y1,'g-',linestyle='--')
l2, = ax2.plot(x1,y2,'b-')
plt.legend(handles=[l1,l2],labels=['电压','电流'],loc='best')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号


# 作图2：SOH-充放电循环次数
plt.figure(2, figsize=(10, 7))
for i in range(len(all_discharge)): 
    # 筛掉第一次循环数据        
    x1 = all_discharge[i]['index'][1:]
    y1 = all_discharge[i]['SOH'][1:]
    #第一行第一列图形
    ax1 = plt.subplot(2,2,1)
    #第一行第二列图形
    ax2 = plt.subplot(2,2,2)
    battery = ['5号电池','6号电池','7号电池','26号电池','27号电池','28号电池']
    if i < 3 :
        plt.sca(ax1)
        plt.plot(x1,y1,label = battery[i])
        plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("SOH(%)", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号       
    else:
        plt.sca(ax2)
        plt.plot(x1,y1,label = battery[i])
        plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("SOH(%)", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号   


# 作图3：电压积分-充放电循环次数
plt.figure(3, figsize=(10, 7))
for i in range(len(all_soh_vi_cycle)): 
    # 筛掉第一次循环数据        
    x1 = all_soh_vi_cycle[i]['index']
    y1 = all_soh_vi_cycle[i]['vi']
    #第一行第一列图形
    ax1 = plt.subplot(2,2,1)
    #第一行第二列图形
    ax2 = plt.subplot(2,2,2)
    battery = ['5号电池','6号电池','7号电池','26号电池','27号电池','28号电池']
    if i < 3 :
        plt.sca(ax1)
        plt.plot(x1,y1,label = battery[i])
        plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号         
    else:
        plt.sca(ax2)
        plt.plot(x1,y1,label = battery[i])
        plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号   


# 作图4：电压积分-SOH
plt.figure(4, figsize=(10, 7))
for i in range(len(all_soh_vi_cycle)): 
    # 筛掉第一次循环数据        
    x1 = all_soh_vi_cycle[i]['SOH']
    y1 = all_soh_vi_cycle[i]['vi']
    #第一行第一列图形
    ax1 = plt.subplot(2,2,1)
    #第一行第二列图形
    ax2 = plt.subplot(2,2,2)
    battery = ['5号电池','6号电池','7号电池','26号电池','27号电池','28号电池']
    if i < 3 :
        plt.sca(ax1)
        plt.scatter(x1,y1,s=12,label = battery[i])
        plt.xlabel("SOH（%）",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号       
    else:
        plt.sca(ax2)
        plt.scatter(x1,y1,s=12,label = battery[i])
        plt.xlabel("SOH（%）",fontproperties = 'KaiTi',fontsize = 15)
        plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
        plt.legend(loc='best')
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号        



'''
# 相关性分析---电压积分与SOH
# pearson系数
#corr_df = df.corr()
df100_70 = soh_vi_cycle.loc[soh_vi_cycle['SOH']>70]
corr_df = df100_70.corr()
corr_df['SOH'].sort_values(ascending=False)
'''


'''    
# 以5号电池为例
no5_data_mat = getBattery(dir_list[0])  
no5_data_df = to_df(no5_data_mat)
no5_discharge = discharge_corr(no5_data_df)


# 5号电池的充电数据
no5_charge = charge_trip(no5_data_df['charge'])[0]
no5_charge_trip_index = charge_trip(no5_data_df['charge'])[1]
no5_charge_trip = get_charge_trip(no5_charge,no5_charge_trip_index)

# 计算电压积分
voltint = calc_VoltIntegral(no5_charge_trip)
soh_vi_cycle = soh_vi_cycle(voltint,no5_discharge)
# 去除掉第一次和最后一次的充电数据
soh_vi_cycle = soh_vi_cycle[1:-1]


# 作图1：电压、电流-时间  #以电池第二次充放电循环数据作为展示
fig1, ax1 = plt.subplots() 
ax2 = ax1.twinx()
x1 = cycle['Time'] 
y1 = cycle['Voltage_measured']
y2 = cycle['Current_measured']
ax1.set_xlabel("时间（s）",fontproperties = 'KaiTi',fontsize = 15)
ax1.set_ylabel("电压（V）",color='g',fontproperties = 'KaiTi',fontsize = 15)
ax2.set_ylabel("电流（A）",color='b',fontproperties = 'KaiTi',fontsize = 15)
plt.xlim([1, 13800])
plt.ylim([-3, 2])
l1, = ax1.plot(x1,y1,'g-',linestyle='--')
l2, = ax2.plot(x1,y2,'b-')
plt.legend(handles=[l1,l2],labels=['电压','电流'],loc='best')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号


# 作图2：SOH-充放电循环次数
fig2 = plt.figure()
x1 = no5_discharge['index'] 
y1 = no5_discharge['SOH']
plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
plt.ylabel("SOH(%)", fontproperties = 'KaiTi',fontsize = 15)
plt.xlim([0, 180])
plt.ylim([60, 100])
plt.plot(x1,y1)
#plt.scatter(x1,y1,s=5)


# 作图3：电压积分-充放电循环次数
fig3 = plt.figure()
x1 = soh_vi_cycle['index']
y1 = soh_vi_cycle['vi']
plt.xlabel("充放电循环次数",fontproperties = 'KaiTi',fontsize = 15)
plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
#plt.plot(x1,y1)
plt.scatter(x1,y1,s=5)


# 作图4：电压积分-SOH
fig4 = plt.figure()
x1 = soh_vi_cycle['SOH']
y1 = soh_vi_cycle['vi']
plt.xlabel("SOH（%）",fontproperties = 'KaiTi',fontsize = 15)
plt.ylabel("电压积分", fontproperties = 'KaiTi',fontsize = 15)
plt.xlim([60, 100])
#plt.ylim([60, 100])
plt.scatter(x1,y1,s=5)


# 相关性分析-pearson系数
#corr_df = df.corr()
df100_70 = soh_vi_cycle.loc[soh_vi_cycle['SOH']>70]
corr_df = df100_70.corr()
corr_df['SOH'].sort_values(ascending=False)
'''






'''挑选最优充电片段'''
'''
# 定义目标函数

# i：第i块电池 -- j：第i块电池的第j次循环 
def demo_func(A,B):
    list_M = []
    for i in range(len(charge_list)):
        list_m = []
        for j in range(len(charge_list[i])):       
            H_AB = calc_VoltIntegral(charge_list[i][j][A:B])
            H_I = calc_VoltIntegral(charge_list[i][j])
            list_m.append((H_I -H_AB)**2)
        e = sum(list_m)/len(list_m)
        list_M.append(e)
    E = sum(list_M)/len(list_M)
    return E
 
    
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


# 使用pso算法寻优
pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)


# 展示结果
plt.plot(pso.gbest_y_hist)
plt.show()
'''






