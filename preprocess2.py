import pandas as pd

data = pd.read_excel("./01_叶片结冰预测/train/15/data1.xlsx")
# print(data.shape)
data = data.iloc[0:50000,:]
# print(data.shape)

data = pd.DataFrame(data)
writer = pd.ExcelWriter('data.xlsx')
data.to_excel(writer)
writer.save()