#複数CSVを1つのDFに結合
import os
import pandas as pd
import numpy as np

path = "/home/nishika_apartment_2021_spring/train"
files = os.listdir(path)
files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
#print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
df_main = pd.DataFrame()
for f in files_file:
	if f.endswith(".csv"): #拡張子を指定
		df_tmp = pd.read_csv(os.path.join(path,f))
		df_main = pd.concat([df_main, df_tmp])

#カラム毎のユニークな要素を一覧で出力
for c in df_main.columns:
		print(c)        
		print(df_main[c].unique())


#指定のカラムのデータ数調査
[[(print(f),df_main[f].nunique())] for f in ["地域","土地の形状","間口","延床面積（㎡）","前面道路：方位","前面道路：種類","前面道路：幅員（ｍ）",]]


#建築年を西暦に
#平成 + 88
#ex) 平成12年　12+88  + 1900 = 2000
#昭和 + 25
#ex) 昭和45年　45+25  + 1900 = 1970
#特定カラムへの関数の適用
def BuildYear(arg_str):
	rec = 0

	byearstr = arg_str
	if byearstr.startswith("昭和"):
		rec = 1900 + 25 + int(byearstr.replace("昭和",""))
	elif byearstr.startswith("平成"):
		rec = 1900 + 88 + int(byearstr.replace("平成",""))
	else: #「戦前」のルート
		rec = 0
	
	return rec

df_tmp[df_tmp.isnull()==True] = '0'
df_tmp["建築西暦"] = df_tmp["建築年"].apply(BuildYear)


#文字列の部分変換
df_tmp["間取りHAN"] = df_tmp["間取りHAN"].str[1:].replace({'K':'000','K+S':'001','D':'100','DK':'110','DK+S':'1111','L':'200','L+S':'201','LD':'220','LD+S':'221','LDK':'220','LDK+K':'222','LDK+S':'221','LK':'210','LK+S':'211'})


#列に型が混在する場合に揃える
df_tmp["面積（㎡）"] = df_tmp["面積（㎡）"].apply(lambda x : int(x))

