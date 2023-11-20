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
