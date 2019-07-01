#==============================================================================#
###1.Merge하기 (kaggle에서 가져온 데이터들)

import numpy as np
import pandas as pd
import os
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
pd_df = pd.read_csv("player_data_deleted.csv")
players=pd_df['name'].unique() #우리가 가진 선수 1904명이 맞게 나오는지 보기 위해서

len(players) #1905가 아닌 1890이 나옴. 중복된 선수가 있는 듯 - 중복된 선수를 찾아 이름 뒤에 표식을 붙이기로 함
player_1904=pd_df['name'] #player data의 이름 column만 가져옴=player_1904
player_1904=player_1904.tolist() #player_1904를 series가 아닌 list로 만듦

player_1904.sort() #오름차순으로 정렬, 중복되는 애들을 찾아내는 코드
prev = None
for i in player_1904:
    if prev == i:
        print("중복된", prev, "이들!")
    prev=i

dup_player = ['Bobby Jones', 'Cedric Henderson', 'Charles Jones', 'Charles Smith', 'Dee Brown', 'Eddie Johnson', 'George Johnson', 'Ken Johnson', 'Marcus Williams', 'Mark Davis', 'Mark Jones', 'Michael Smith']
#중복되는 데이터들은 지웠음
players_list=np.array(players.tolist())
whyme_list=np.array(whyme.tolist()) #np.array를 list로 변환했다
listC = list(set(players_list). difference(whyme_list)) #둘중 겹치지 않는 걸 확인 반대는 intersection임
#그랬더니 pd_df는 중복된거 지워서 1878인데 merge된건 1809임. 그래서 69명 찾았음 누락된거.


os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
pd_df = pd.read_csv("player_data_1112.csv") #중복되는 애들 제거한거
#알고보니 ss_df 파일에는 word 3개 이상으로 이뤄진 선수들이 전부 다 삭제되어 있었다. 그래서 일일이 pd_df에서 원래 이름을 찾아서 바꿔주었음.
ss_df = pd.read_csv("Seasons_Stats_3word_namegive.csv")
merge_df=pd.merge(pd_df, ss_df, on='name')
players=pd_df['name'].unique() #1904명에서 중복되는 친구들 다 제거하니 1878명의 선수가 남았다
whyme=merge_df['name'].unique() #69 #1809
#최종적으로 merge 된 파일이 merge_df 파일임. 총 1878명의 선수들.

#==============================================================================#
###2.추가적으로 수집한 데이터들도 Merge하기 (HoF, 검색수 등)
#검색수 합치기
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
mg_df = pd.read_csv("merged_df.csv")
sc_df = pd.read_csv("search_list.csv")
merge_df=pd.merge(mg_df, sc_df, on='name')
merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/merged_df_addsearch.csv', sep=',', na_rep='NaN')
#검색수까지 합친 파일이 addsearch

#이제 HoF 합치기
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
mg_df = pd.read_csv("merged_df_addsearch.csv")
hof_df = pd.read_csv("player_HoF.csv")
merge_df=pd.merge(mg_df, hof_df, on='name')
merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/merged_df_addhof.csv', sep=',', na_rep='NaN')
#검색수와 hall of fame 여부 (0/1)까지 합친 파일이 addhof.csv

#이제 우승수 합치기
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
mg_df = pd.read_csv("merged_df_addhof.csv")
win_df = pd.read_csv("player_final_win_num.csv")
merge_df=pd.merge(mg_df, win_df, on='name')
merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/merged_df_addwin.csv', sep=',', na_rep='NaN')
#검색수와 hall of fame 여부 (0/1), 그리고 우승 횟수까지 합친 파일이 merged_df_addwin.csv


#==============================================================================#
###3. 선수별 스탯 데이터 전처리 (MAX, SUM)
#이제 TOT 삭제랑 이름으로 sum하자
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
mg_df= pd.read_csv("presum_df.csv") #addwin에서 더해지면 안되는 column들 지운 파일
presum_df=mg_df[~mg_df.select_dtypes(['object']).eq('TOT').any(1)] #presum_df에서 TOT있는 열 다 날림
sum_df = presum_df.groupby("name").sum() #이름으로 group 지어서 총 row가 1878개 나옴
sum_df=sum_df.drop(['Unnamed: 0'], axis=1) #필요없는 column 삭제 (unnamed)
sum_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/sum_df.csv', sep=',', na_rep='NaN')
#addwin에서 TOT삭제하고 이름으로 sum한 파일이 (더해지면 안되는 column들은 삭제해서 이후 추가해줘야함 hof같은거) sum_df임

#이제 이름으로 그룹바이하구 max하자
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
premax_df= pd.read_csv("premax_df.csv") #addwin에서 더해지면 안되는 column들 지운 파일
max_df = premax_df.groupby("name").max()
max_df=max_df.drop(['Unnamed: 0'], axis=1) #필요없는 column 삭제 (unnamed)
max_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/max_df.csv', sep=',', na_rep='NaN')

#sum이랑 max랑 합치기 전 sum의 column들 이름에 일괄적으로 _sum을 붙임
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
sum_df= pd.read_csv("sum_df.csv") #ㄴ
columns=sum_df.columns
sum_df.rename(columns = lambda x : x+"_sum", inplace = True) #일괄적으로 뒤에 _sum 해주기
sum_df = sum_df.rename(columns={'name_sum':'name'}) #name은 _sum 삭제
sum_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/sum_editname_df.csv', sep=',', na_rep='NaN')

#max랑 sum을 합쳤다 (sum은 _sum으로 컬럼 이름 수정했음)
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
max_df= pd.read_csv("max_df.csv")
sum_df= pd.read_csv("sum_editname_df.csv")
merge_df=pd.merge(max_df, sum_df, on='name')
merge_df=merge_df.drop(['Unnamed: 0'], axis=1) #필요없는 column 삭제 (unnamed)
merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/merge_maxsum_df.csv', sep=',', na_rep='NaN')

#merge_maxsum_df에 기존 삭제했던 column들 (hof등등...)을 더해준다
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
merge_old_df= pd.read_csv("merge1.csv")
merge_df= pd.read_csv("merge_maxsum_df.csv")
merge_deleted_df=merge_old_df.drop(['year_start', 'year_end', 'position', 'height',
       'weight', 'birth_date', 'college', 'Year', 'Age', 'Tm', 'G',
       'PER', 'TS%', 'WS', 'TRB', 'AST', 'PTS'], axis=1) #필요없는 column 삭제 (unnamed)
merge_deleted_df=merge_deleted_df.drop(['Unnamed: 0'], axis=1) #unnamed삭제
merge_deleted_df=merge_deleted_df.groupby("name").first(check_identical=True) #이름으로 묶고 첫번째 것만 남기기
all_merge_df=pd.merge(merge_deleted_df, merge_df, on='name')
all_merge_df=all_merge_df.drop(['Unnamed: 0'], axis=1) #unnamed삭제
all_merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/all_merge_df.csv', sep=',', na_rep='NaN')
#최종적으로 더한 값. mix랑 sum이랑 sum되면 안되서 삭제했던 것들까지 다 합친 값. 1878개의 row

#height값에서 누락된 3글자 리스트 찾기
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
yes_df= pd.read_csv("Seasons_Stats_3word_namegive.csv")
no_df= pd.read_csv("Seasons_Stats_deleted.csv")
yes_name=yes_df['name'].unique()
no_name=no_df['name'].unique()
yes_list=np.array(yes_name.tolist())
no_list=np.array(no_name.tolist())
dup_list= list(set(yes_list). difference(no_list)) #둘중 겹치지 않는 걸 확인 반대는 intersection임
print(dup_list)
#참고-3글자 선수들 리스트 ['Micheal Ray Richardson', 'Eddie Lee Wilkins', 'Joe Barry Carroll', 'Keith Van Horn', 'Jan Van Breda Kolff', 'World B. Free', 'Horacio Llamas Grey', 'Hot Rod Williams', 'Juan Carlos Navarro', 'Peter John Ramos', 'Vinny Del Negro', 'Jo Jo English', 'Billy Ray Bates', 'Logan Vander Velden', 'Nick Van Exel', 'Jo Jo White']
#그렇게 해서 3글자 해준게 name height_df_namegive

#all_merge에 height과 weight 더하기
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
ht_df= pd.read_csv("height_df_namegive.csv")
merge_df= pd.read_csv("all_merge_df.csv")
ht_merge_df=pd.merge(ht_df, merge_df, on='name')
ht_merge_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/ht_merge_df.csv', sep=',', na_rep='NaN')
#ht_merge_df 가 현재 우리의 최종

#==============================================================================#
###4.결측치 제거와 데이터 정규화
'''
['name', 'height', 'Pos', 'search_num', 'HoF', 'final_win_num', 'G','PER', 'WS', 'TRB', 'AST', 'PTS', 'G_sum', 'PER_sum', 'WS_sum','TRB_sum', 'AST_sum', 'PTS_sum']
중에서 정규화가 필요한 column: height, search_num, final_win_num, G, PER, WS, TRB, AST, PTS, G_sum, WS,sum, TRB_sum, AST_sum, PTS_sum
정규화가 필요하지 않은 column: Pos, HoF(Y값)
Pos는 One Hot Encoding 해야함
'''

#z-정규화를 하려다가 PER column 에서 에러가 나서 왜그런가 하고 봤더니 per에 결측치가 있었음. 그래서 결측치 있나 먼저 봤구...nan삭제해줌(nandie). 그리구 comma도 삭제해줌 (alldie)
#이제 정규화!
final_df= pd.read_csv("final_df_alldie.csv")
from sklearn.preprocessing import StandardScaler
z_scaler = StandardScaler().fit(final_df[['height', 'search_num', 'final_win_num', 'G','PER', 'WS', 'TRB', 'AST', 'PTS', 'G_sum', 'PER_sum', 'WS_sum','TRB_sum', 'AST_sum', 'PTS_sum']])
z_mat = z_scaler.transform(final_df[['height', 'search_num', 'final_win_num', 'G','PER', 'WS', 'TRB', 'AST', 'PTS', 'G_sum', 'PER_sum', 'WS_sum','TRB_sum', 'AST_sum', 'PTS_sum']])
auto_df=pd.DataFrame(z_mat)
# column header 추가해주기
auto_df.columns=['height', 'search_num', 'final_win_num', 'G','PER', 'WS', 'TRB', 'AST', 'PTS', 'G_sum', 'PER_sum', 'WS_sum','TRB_sum', 'AST_sum', 'PTS_sum']
auto_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/afterz_df.csv', sep=',', na_rep='NaN')

#이제 Hof(Y값),이름,Pos 처리해주면 된다. 이것들까지 더한게 afterz_add_df

#이제 onehotencoding!
#POS를 9개로 수정한 뒤 onehotencoding
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
add_df= pd.read_csv("n_afterz_add_df.csv")
add_df['Pos'] = add_df['Pos'].map({'PG' : 'PG', 'SG' : 'SG', 'SF' : 'SF', 'PF' :'PF', 'C' : 'C', 'PG-SG' :'PG-SG', 'SG-PG' :'PG-SG' , 'SG-SF' : 'SG-SF', 'SF-SG' : 'SG-SF', 'SF-PF' : 'SF-PF', 'PF-SF' : 'SF-PF' , 'PF-C' : 'PF-C', 'C-PF' : 'PF-C'})
add_df=pd.get_dummies(add_df)
add_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/halloffame_df.csv', sep=',', na_rep='NaN')
#position을 9개로 수정하고 z-정규화 & 원핫인코링전처리 완료. 최종파일이 halloffame


import sklearn
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
hof_df= pd.read_csv("halloffame_df.csv")
hof_df=hof_df.drop(['Unnamed: 0'], axis=1) #필요없는 column 삭제 (unnamed)
hof_df.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/halloffame_df.csv', sep=',', na_rep='NaN')

#==============================================================================#
###5.모델돌리기 (1)Decision Tree
#데이터 불러오고 Decision Tree
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
hof_df= pd.read_csv("halloffame_df.csv")
X = np.array(hof_df.iloc[:,0:24])
y = np.array(hof_df['HoF'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print(tree.score(X_train,y_train))
print(tree.score(X_test,y_test))
#결과값은 result_decision tree에 이따! 1.0, 0.97정도로 꽤 높게 나옴 그런데.....

#==============================================================================#
###6.모델돌리기 (2)confusion matrix & F_score
#우리는 class의 불균형이 심해서 accuracy로는 정확도가 왜곡될 염려가 있기때문에 데이터 불러오고 F_score 해볼 것임! sklearn3
os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
hof_df= pd.read_csv("halloffame_df.csv")

X = np.array(hof_df.iloc[:,0:24])
y = np.array(hof_df['HoF'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

hof_df['HoF'].unique()
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred, labels=[0,1])
print(confusion)
#결과값은 confusion_matrix result로 캡쳐
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#결과값은 score로 캡쳐
print("accuracy:", accuracy_score(y_test, pred))
print("precision:", precision_score(y_test, pred, average='macro'))
print("recall:", recall_score(y_test, pred, average='macro'))
print("f1_score:", f1_score(y_test, pred, average='macro'))

from sklearn.metrics import classification_report
results = classification_report(y_test, pred, labels=[0,1])
print(results)
#결과값은 classification_report로 캡쳐

#==============================================================================#
###7. 모델돌리기 (3)oversampling 후 confusion matrix & F_score
#oversampling!!!!!
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
hof_df= pd.read_csv("halloffame_df.csv")
hof_df=hof_df.drop(['Unnamed: 0'], axis=1) #필요없는 column 삭제 (unnamed)

y = hof_df.HoF
X = hof_df.drop('HoF', axis=1)

clf_0 = LogisticRegression().fit(X,y)

pred_y_0 = clf_0.predict(X)

from sklearn.utils import resample
hof_df_majority = hof_df[hof_df.HoF==0]
hof_df_minority = hof_df[hof_df.HoF==1]

hof_df_minority_upsampled = resample(hof_df_minority,replace=True,n_samples=1821,random_state=123)

hof_df_upsampled = pd.concat([hof_df_majority, hof_df_minority_upsampled])
hof_df_upsampled.to_csv('/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project/hof_df_upsampled.csv', sep=',', na_rep='NaN')
#upsampling한 데이터를 hof_df_upsampled로 수정했다.

#새로운 데이터를 가지고 다시 모델 돌려보기
#데이터 비대칭이 있다는걸 확인했으니 이를 해결할 undersampling & oversampling 해보자!
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.chdir(r"/Users/hanameee/Desktop/학교/2018-1학기/데이터마이닝/term_project")
hof_df= pd.read_csv("hof_df_upsampled.csv")
X = np.array(hof_df.iloc[:,0:24])
y = np.array(hof_df['HoF'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

hof_df['HoF'].unique()
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred, labels=[0,1])
print(confusion)
#결과값은 confusion_matrix result로 캡쳐
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#결과값은 score로 캡쳐
print("accuracy:", accuracy_score(y_test, pred))
print("precision:", precision_score(y_test, pred, average='macro'))
print("recall:", recall_score(y_test, pred, average='macro'))
print("f1_score:", f1_score(y_test, pred, average='macro'))

from sklearn.metrics import classification_report
results = classification_report(y_test, pred, labels=[0,1])
print(results)

#다시 돌려서 어느정도 f1 score도 유의미한 모델을 만들었당.
#참고 https://elitedatascience.com/imbalanced-classes

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:05:45 2018

@author: user
"""
"""
Salary ratio 와 Salary data set의 전처리 코드 및 모델 학습, 평가 코드
"""

import os
import numpy as np
import pandas as pd
import sklearn

os.chdir(r"C:\Users\user\Desktop")
salary_df = pd.read_csv('player_salary_df.csv')
salary_df.head


# 0이 있는 row를 다 없애는 코드 >>>> salary data 에는 0인 element가 없기때문에 사용 가능
salary1_df=salary_df[~salary_df.select_dtypes(['integer']).eq(0).any(1)]
salary1_df

# player 기준으로 groupby 한 후 각 player 들의 평균 연봉액만 남긴다.
maxsalary_df = salary1_df.groupby("Player").mean()
maxsalary_df

# salary data set에서 필요없는 변수 column 지운다
maxsalary_df.drop(['Year','Team','YearEnd'], axis=1)
realsalary_df=maxsalary_df.drop(['Year','Team','YearEnd'], axis=1)
realsalary_df

realsalary_df.to_csv("realsalary_df.csv",index=True)

# 연봉 데이터로 연봉 인상률 컬럼을 추가한다 (for 문 이용)
df = pd.read_csv('realsalary_df.csv', na_values= 'Unknown')
df.head()

df.dropna(inplace=True)
s = []
f_idx = []
for i in range(len(df)):
    try:
        target = df['Salary'][i]
        n_target = float(target.replace(',',''))
        print(n_target)
        s.append(n_target)
    except:
        s.append(np.nan)
        f_idx.append(i)

f_idx
df['n_salary'] = s
df.head()

ratio_list = []
for i in range(len(df)):
    print(i)
    if i == 0 :
        ratio = 0
    elif i == len(df)-1:
        ratio = 0
    elif df['Player'][i] == df['Player'][i+1]:
        ratio = (df['n_salary'][i+1] - df['n_salary'][i]) / df['n_salary'][i]
    elif df['Player'][i] != df['Player'][i+1]:
        ratio = 0

    print(ratio)
    ratio_list.append(ratio)


df['ratio'] = ratio_list

df.to_csv('salary_ratio_1.csv', index=False)



#1. 전처리 시작!__________________________________________________________________

os.chdir(r"/Users/heewoo/Downloads")
HOF = pd.read_csv('halloffame_df.csv')
HOF.head()


# 1) salaryratio_mean과 HOF 팀 파일 merge
#salarqy ratio_mean: 각 player의 mean salary ratio 구하기
salaryratio_mean = pd.read_csv('salary_ratio_1.csv')
salaryratio_mean.head()
salaryratio2 = salaryratio_mean.groupby('Player').mean()
salaryratio3=salaryratio2.fillna(0)
salaryratio3.head()
salaryratio3.to_csv("salaryratio_2.csv",index=True)

# 2) salary ratio와 player 데이터 merge
salaryratio4 = pd.read_csv('salaryratio_2.csv')
salaryratio4.head()
salaryratio = pd.merge(HOF,salaryratio4, on='Player')
salaryratio.head()
salaryratio.to_csv("salaryratio_data.csv",index=False)

# 전처리 완료_____________________________________________________________________



#1. Minmax Scaler for y = salary_______________________________________________
import os
import sklearn
import numpy as np
import pandas as pd 

os.chdir(r"/Users/heewoo/Downloads")
salary_data = pd.read_csv('salary_real.csv')



# 1) 데이터 결과를 높이자... minmax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(salary_data[['height','search_num','final_win_num','G_sum','PER_sum','WS_sum','TRB_sum','AST_sum','PTS_sum','HoF']])
mat = scaler.transform(salary_data[['height','search_num','final_win_num','G_sum','PER_sum','WS_sum','TRB_sum','AST_sum','PTS_sum',
                                  'HoF']])
minmax = salary_data
minmax.to_csv("minmax.csv",index=False)


salary_data['n_height']=mat[:,0:1]
salary_data['n_search_num']=mat[:,1:2]
salary_data['n_final_win_num']=mat[:,2:3]
salary_data['n_G_sum']=mat[:,3:4]
salary_data['n_PER_sum']=mat[:,4:5]
salary_data['n_WS_sum']=mat[:,5:6]
salary_data['n_TRB_sum']=mat[:,6:7]
salary_data['n_AST_sum']=mat[:,7:8]
salary_data['n_PTS_sum']=mat[:,8:9]
minmax = salary_data
minmax.to_csv("minmax.csv",index=False)
#______________________________________________________________________________

# 1-1 Linear Regression
salary_data2 = pd.read_csv('minmax.csv')
X = np.array(salary_data2.iloc[:, 0:10])
y = np.array(salary_data2['n_salary'])

# 1) data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state=0)
print("X_train 크기:",X_train.shape)
print("y_train 크기:", y_train.shape)
print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)
print(X_train)

# 2) Model 돌리기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("훈련데이터 결과:", lr.score(X_train, y_train))
print("검증데이터 결과:", lr.score(X_test, y_test))

# 3) coefficient 와 intercept의 확인
print("intercept:", lr.intercept_) #y intercept
print("coefficient:", lr.coef_) #b 값: x 변수가 y 변수에 미치는 영향

f_names = salary_data2.columns[0:10]#column 개수
f_coef = lr.coef_[0:10]
for name, coef in zip(f_names, f_coef):
    print(name, coef)
#그 값에 따라 가장 영향을 많이 끼치는 x를 알 수 있다.
#______________________________________________________________________________
    
# 1-2 Decision Tree

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor (max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("훈련 데이터 결과:", tree.score(X_train,y_train))
print("검증 데이터 결과:", tree.score(X_test, y_test))
print("\n")

# 1) tree visualization

from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import graphviz
dot_data = export_graphviz(tree, out_file = None,
                           class_names=salary_data['n_salary'].unique(),
                           feature_names=salary_data.columns[0:10],
                           filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('minmax_salary_tree.png')
Image(graph.create_png())

#______________________________________________________________________________


# 1-3 Random Forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=15,random_state=0)
forest.fit(X_train, y_train)
# 1) rf r^ 값 
print("RF:")
print("훈련 데이터 결과:", forest.score(X_train, y_train))
print("검증 데이터 결과:", forest.score(X_test, y_test))
print("\n")

# 2) 최적의 n_estimatiors 구하기
k=[1,3,5,10,15,20]
for i in k :
    forest = RandomForestRegressor(n_estimators = i, random_state=0)
    forest.fit(X_train, y_train)    
    print("%s인경우"%i)
    print("훈련 데이터 결과:", forest.score(X_train,y_train))
    print("검증 데이터 결과:", forest.score(X_test,y_test))
#______________________________________________________________________________
    
    
# 1-4 KNN 모델
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train,y_train)
print("KNN:")
print("훈련 데이터 결과:", knn.score(X_train, y_train))
print("검증 데이터 결과:", knn.score(X_test, y_test))
print("\n")

# 1) KNN 최적 모델 탐색  포문 (1,3,10,30,50)
x = [1, 3, 10, 30, 50]
for n_neighbor in x:
    print(n_neighbor)
    knn = KNeighborsRegressor(n_neighbors = n_neighbor)
    knn.fit(X_train,y_train)
    print("훈련 데이터 결과:", knn.score(X_train, y_train))
    print("검증 데이터 결과:", knn.score(X_test, y_test))
#______________________________________________________________________________
    
    
# 2-1 Regression Summary
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import statsmodels.api as sm
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=50, n_features=2, cluster_std=5.0,
                  centers=[(0,0), (2,2)], shuffle=False, random_state=12)
logit_model = sm.Logit(y, sm.add_constant(x)).fit()
print (logit_model.summary())
 
# 2-2 Regression Metrics

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print(pred[:10])
print(y_test[:10])

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("MAE:", mean_absolute_error(y_test,pred))
print("MSE:", mean_squared_error(y_test,pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,pred)))
print("R^2:", r2_score(y_test,pred))

# 2-3 Dummy Regressor
from sklearn.dummy import DummyRegressor
dummy=DummyRegressor(strategy="mean")
dummy.fit(X_train,y_train)
d_pred = dummy.predict(X_test)
d_pred[:18]
print("MAE:", mean_absolute_error(y_test,pred))
print("MSE:", mean_squared_error(y_test,pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,pred)))
#______________________________________________________________________________


# 3-1 feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

select=SelectFromModel (RandomForestRegressor(n_estimators=10,random_state=0),threshold="mean")
select.fit(X_train,y_train)
select.get_support()
select_idx=np.where(select.get_support() ==True)
select_idx
zscore_salary_data.columns[select_idx]
rf = RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X_train,y_train)
print("전체 변수 사용:", rf.score(X_test,y_test))
X_train_selected = select.transform(X_train)
X_test_selected= select.transform(X_test)
rf.fit(X_train_selected,y_train)
print("선택 변수 사용:", rf.score(X_test_selected, y_test))


# 3-2 Feature Selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
select=RFE(RandomForestRegressor(n_estimators=10, random_state=0),n_features_to_select=1)
select.fit(X_train, y_train)
sum(select.get_support())
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
total_score = RandomForestRegressor(n_estimators = 10,random_state=0).fit(X_train,y_train).score(X_test,y_test)
rfe_score = RandomForestRegressor(n_estimators = 10, random_state=0).fit(X_train_rfe,y_train).score(X_test_rfe,y_test)
print("total_score:", total_score)
print("rfe_score:", rfe_score)
#______________________________________________________________________________


