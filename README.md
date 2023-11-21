# YOLO-SSL-cont

## How to run code(Mode1 YOLO背景執行)
終端機路徑執行路徑
```
./yolossl
```
權重放置路徑(兩個都需要)
```
./yolossl/weights
```
```
./yolossl/yolov7/weights
```
權重名稱
```
YOLOSSL.pt
```
### 首先載入偵測模型(不中斷執行) 
需要打開第一個終端機後執行指令
```
python yolov7/detectcont.py
```

### 切割圖片後偵測以及合併
打開第二個終端機後執行指令

```
python testcont.py ../configs/testPCBsl.yaml
```

### 需要偵測圖片路徑
```
rawdata/
```

### 偵測結果圖片路徑
```
results/test
```

## How to run code(Mode2 YOLO-SSL單次執行)
### 偵測以及前後處理執行指令
```
python test.py ../configs/testPCBsl.yaml
```

### 需要偵測圖片路徑
```
rawdata/
```

### 偵測結果圖片路徑
```
results/test
```

## config中的yaml file參數設定
### name_file_name
檔案需要放在yolosl資料夾內，檔名為 xxx.yaml，以下為yaml file範例
```
train: ship/train/images
val: ship/valid/images
nc: 2
names: ['-', 'Ship']
```

### object_names: 
對應上面names的分類
```
- '-'
- 'Ship'
```

### sealand
如果為 True 則會對切割後的圖片進行海陸分離

### test_im_dir
要偵測的圖片資料夾位置

### sliceHeight and Width
切割後的圖片大小
### slice_overlap
每一個切割相鄰圖片的重疊比例
