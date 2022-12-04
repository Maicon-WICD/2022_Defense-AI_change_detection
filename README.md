# 국방 AI 경진대회 코드 사용법
- wicd팀, 정원국, 신호준, 안정수, 전지용
- 닉네임 : wonguk, inthecode, carina, DLHjjy


# 핵심 파일 설명
  - 노이즈데이터 제거 코드 : `./delNoise.py`
  - 학습 데이터 경로: `./data/train/`
  - 학습 메인 코드: `./train.py`
  - 테스트 메인 코드: `./predict.py`
  - 테스트 이미지, 마스크 경로: `./data/test/`
  - 테스트 결과 이미지 경로: `./results/pred/`

## 코드 구조 설명

- 본 대회에서 제공하는 baseline 코드를 기반으로 segmentation models pytorch를 사용하여 학습 및 테스트했습니다.
    - 최종 사용 모델 : segmentation models pytorch에서 제공하는 DeepLabV3+ 
    - 최종 사용 인코더 : segmentation models pytorch에서 제공하는 timm-regnetx_064 
    - 최종 사용 웨이트 : segmentation models pytorch에서 제공하는 imagenet
- baseline 코드에 focaloss, clahe scaler, agumentation을 추가하고 stride를 변경 할 수 있도록 코드를 작성했습니다.
    - focal loss 추가
    ```
    ./modules/losses.py (line 87-110)
    ```
    - clahe scaler 추가
    ```
    ./modules/losses.py (line 38-49)
    ```
    - albumentations을 사용한 agumentation 추가
    ```
    ./modules/datasets.py (line 43-76)
    ```
    - stride 변경 코드 추가:
    ```
    ./modules/train.py (line 99)
    ```


- **최종 제출 파일 : submit_file/mask.zip**
- **학습된 가중치 파일 : final_weight/model.pt**

## 주요 설치 library
- albumentations==1.3.0
- matplotlib==3.5.3
- opencv-python==4.6.0.66
- numpy==1.21.5
- pandas==1.3.5
- pickle5==0.0.12
- Pillow==9.2.0
- PyYAML==6.0
- scikit-learn==1.0.2
- scikit-image==0.19.3
- segmentation-models-pytorch==0.3.0
- torch==1.13.0 
- torchvision==0.14.0
- tqdm==4.64.1
- typing_extensions==4.3.0


# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치
    ```
    conda create -y -n change python=3.7
    conda activate change
    conda install -y pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install -y scikit-learn==1.0.2
    pip install pickle5==0.0.12
    pip install PyYAML==6.0
    pip install opencv-python==4.6.0.66
    pip install albumentations==1.3.0
    pip install matplotlib==3.5.3
    pip install pandas==1.3.5
    pip install tqdm==4.64.1
    pip install segmentation-models-pytorch==0.3.0
    '''
# 학습 실행 방법

  - noise 데이터 제거 코드 실행
  ```
  python ./delNoise.py
  ```

  - 학습 코드 실행
  ```
  python ./train.py
  ```

# 테스트 실행 방법

  - 테스트 config 수정
  ```
  ./config/predict에서 train_serial 값을 train한 결과의 폴더이름으로 변경
  ```

  - 테스트 코드 실행
  ```
  python ./predict.py
  ```

