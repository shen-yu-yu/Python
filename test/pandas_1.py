import pandas as pd

# 读取数据
chipo = pd.read_csv('../data/chipotle.csv', sep='\t')

# 查看数据基本信息
# print(chipo.head(10))
print(chipo.columns.tolist())
# print(chipo.index)

#被下单数最多商品(item)是什么?
print(chipo[['item_name', 'quantity']].groupby(by=['item_name']).sum().sort_values(by=['quantity'], ascending=False))