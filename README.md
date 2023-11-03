# <h1 align= 'center'>國立陽明交通大學光電工程學系碩士論文 -- 程式部分</h1>
# Thesis title

### 智慧功能性近紅外光光譜術應用於思覺失調症與雙向情緒障礙症： 可解釋人工智慧演算法之可行性評估

### Intelligent functional near-infrared spectroscopy for schizophrenia and bipolar disorder: Feasibility assessment of an explainable artificial intelligence algorithm

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
└ schizo_control_LSTM.ipynb
```
# Files and Environment
* ### Data_preprocessing.ipynb
  * 資料前處理，包含資料清潔、訊號合併以及正規化等
  * 使用**Python 3.9(anaconda) + Visual Studio Code**，並未使用GPU
* ### schizo_control_LSTM.ipynb
  * 模型訓練，包含Dataset、Dataloader、model的編寫，以及訓練方式和結果可視化，並且還有可解釋人工智慧的運用
  * 使用 **Kaggle**提供之伺服器，搭配提供之P100做訓練
 
# Prepared dataset
訓練資料的製作主要在`Data_preprocessing.ipynb`檔案中，我主要做了四個處理:

1. 定義正規化函數
2. 定義資料處理函數以及訊號合併函數 - No Minmax processing
3. 定義資料處理函數以及訊號合併函數 - Minmax processing
