# <h1 align= 'center'>國立陽明交通大學光電工程學系碩士論文</h1>
# Thesis title

### 智慧功能性近紅外光光譜術應用於思覺失調症與雙向情緒障礙症： 可解釋人工智慧演算法之可行性評估

Intelligent functional near-infrared spectroscopy for schizophrenia and bipolar disorder: Feasibility assessment of an explainable artificial intelligence algorithm

# Abstract
目前在臨床精神疾病的診斷中，主要仰賴醫生的專業經驗以及使用量表進行評估。然而，
由於不同醫師的專業經驗可能有所不同，且量表會有病人的主觀部分在，從而使得病患無法獲
得客觀的診斷方式。此外，思覺失調症、雙向情緒障礙症與憂鬱症等精神疾病在臨床上的特徵
有時會相似，從而難以正確的診斷疾病，而這些疾病的治療方法也存在顯著的差異。如果無法
在早期準確進行診斷，將可能導致患者的治療成本上升以及病況的惡化。
為了解決這一問題，本研究對健康人、思覺失調症患者和雙向情緒障礙症患者在進行語意
流暢度測驗時，通過使用近紅外光光譜術來量測患者腦部血流的變化。並且結合深度學習和可
解釋的人工智能技術協助醫生進行診斷。在對控制組以及患有思覺失調症的病患的分類上，模
型在訓練集和測試集上的準確度分別達到了0.941 和0.833。同樣地，在控制組和患有雙向情
緒障礙症的病患分類中，我們的模型在訓練集和測試集上的準確度分別為0.936 和0.909。在
二階段的三分類分類中，模型在控制組和疾病組的訓練準確率和測試準確率分別為0.958 和
0.944，而在思覺失調症和雙向情緒障礙症的分類中，訓練準確率和測試準確率分別為0.916
和0.909。
這些結果清楚地表明，我們的實驗方法能夠有效地利用血氧資訊來區分精神疾病，並證實
了我們提出的方法的可行性。這一研究能輔助醫生增加診斷的準確性，並降低治療成本以及提
升患者預後。

**關鍵字：功能性近紅外光光譜術、思覺失調症、雙向情緒障礙症、語意流暢度測驗、深度學
習、可解釋AI**

# Repository structure
```
┌ README.md
├ Data_preprocessing.ipynb
├ schizo_control_LSTM.ipynb
└ image.png
```
# Files and Environment
* ### Data_preprocessing.ipynb
  * 資料前處理，包含資料清潔、訊號合併以及正規化等
  * 使用**Python 3.9(anaconda) + Visual Studio Code**，並未使用GPU
* ### schizo_control_LSTM.ipynb
  * 模型訓練，包含Dataset、Dataloader、model的編寫，以及訓練方式和結果可視化，並且還有可解釋人工智慧的運用
  * 使用 **Kaggle**提供之伺服器，搭配提供之P100做訓練
 
# Prepared dataset
程式在`Data_preprocessing.ipynb`檔案中，我主要做了四個處理:

1. 定義正規化函數
   如下:
   ```python
   def minmax(df_temp):
       dfs =  (df_temp - df_temp.min())/(df_temp.max() - df_temp.min())
       return dfs
   ```
3. 定義資料處理函數以及訊號合併函數 - No minmax processing

   參考[期刊論文](https://www.frontiersin.org/articles/10.3389/fpsyt.2021.655292/full)，將訊號從52 channel合併至2 channel的示意程式
   ```python
   region_1 = df[['CH25', 'CH26', 'CH27', 'CH28', 'CH36', 'CH37', 'CH38', 'CH46' , 'CH47', 'CH48', 'CH49']].mean(axis=1)
   region_2 = df[['CH22', 'CH23', 'CH24', 'CH32', 'CH33', 'CH34', 'CH35', 'CH43', 'CH44', 'CH45', 'CH29', 'CH30', 'CH31', 'CH39', 'CH40', 'CH41', 'CH42', 'CH50', 'CH51', 'CH52']].mean(axis=1)
   dff = pd.concat([region_1, region_2], axis=1)
   ```
5. 定義資料處理函數以及訊號合併函數 - With minmax processing
6. 套用函數，並將製作結果存成.npy檔

# Deep learning modeling
建模程式在`schizo_control_LSTM.ipynb`檔案中，我主要做了五個處理:

1. 讀取各族群資料並合併成Dataframe
2. 製作**Dataset**與**Dataloder**以供後續作訓練

   Dataset中主要定義讀取檔案的方式，以及套用transform於讀入之資料
   ```python
   class CustomImageDataset(Dataset):
       def __init__(self, annotations_file, transform=None):
    
           self.dataframe = annotations_file
           self.transform = transform
           
    
       def __len__(self):
           return len(self.dataframe)
    
       def __getitem__(self, idx):
    
           img_path = self.dataframe.iloc[idx, 0]        
           np_araay = np.transpose(np.load(img_path.replace('\\', '/')))               
           label = self.dataframe.iloc[idx, 1]
    
           if self.transform:
               np_araay = self.transform(np_araay)
                      
           return np_araay, label, img_path 
   ```
3. 製作模型
   本研究使用的是單層的LSTM模型，結構如下
   ```python
   class Network(nn.Module):
       def __init__(self, pool = 6, hid_lay=1, fc1= 256, outlay = 64):
           super(Network, self).__init__()
           # LSTM
           self.outlay = outlay
           self.LSTM1 = nn.LSTM(1251, self.outlay, hid_lay)#, bidirectional=True)
           
           # FC        
           self.fc1num = fc1
           self.fc1 = nn.Linear(2*self.outlay, self.fc1num)
           self.fc2 = nn.Linear(self.fc1num, 64)
           self.fc3 = nn.Linear(64, 1)
           # sigmoid
           self.soft = nn.Sigmoid()
    
           self.drop = nn.Dropout(0.3)
    
    
       def forward(self, input1):
           output = self.LSTM1(input1)[0]
           output = output.view(-1, 2*self.outlay)
           # FC  forward
           con = self.fc1(output)
           con = F.relu(self.drop(con))
           con = self.fc2(con)   
           con = self.fc3(con)
           con = self.soft(con)
    
           return con
   ```
5. 使用`Optuna`尋找最佳參數
   其使用需定義一個訓練函數，以回傳每次訓練結果
   ```python
   def object_fun(trial):
       ...
       ...
       return train_accuracy
   ```
   並且建立`study`去自動尋找最佳參數
   ```python
   # Define sample
   sampler = optuna.samplers.TPESampler(seed=10)
   study = optuna.create_study(storage="sqlite:///cnn_npy_52_channel.db", study_name="mystudy", direction='maximize', sampler=sampler)
   study.optimize(object_fun, n_trials=200)
   ```
7. 套用最佳參數，訓練預測並用Matplotlib及混淆矩陣可視化結果

# Explainable AI - Intergated Gradients
可解釋AI程式位於`schizo_control_LSTM.ipynb`檔案中，我主要做了三個處理:

1. 使用`Captum`套包中的`IntegratedGradients`，將模型與訊號丟入，之後得出`attributions`即為模型所關注的區域

   ```python
   # IntegratedGradients
   df_cuba = pd.DataFrame()
   attributions, delta = ig.attribute(input_data, target=0, return_convergence_delta=True)
   for nums, att in enumerate(attributions):
       df_cuba[f'{nums}_Region_0'] = att.cpu()[0]
       df_cuba[f'{nums}_Region_1'] = att.cpu()[1]
   ```    
3. 使用`rolling apply`數值平滑，以方便畫圖及觀察
   ```python
   minmax(df_cuba.rolling(50).mean().bfill())
   ```
5. 在背景畫出模型主要觀察的位置，結果圖如下，顏色越紅為越重要之區域:
![image](https://github.com/JulianLee310514065/Master_thesis/blob/main/image.png)
