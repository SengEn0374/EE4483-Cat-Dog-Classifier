# # using csv reader
# import csv
#
# fp = '../cifar-10/trainLabels.csv'
# f = open(fp , 'r')
# csvf = csv.reader(f)
# header = next(csvf)
# imgs = []
# for row in csvf:
# 	imgs.append(row)
# print(header)
# print(imgs)
# f.close()


import pandas as pd

data = pd.read_csv("../cifar-10/trainLabels.csv")
# print(data) # cifar 10 image id ('id') starts from 1, csv auto id starts from 0
# print(type(data.id[0])) # int64

