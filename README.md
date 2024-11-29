# 제목 없음

# README: 고양이와 강아지 분류를 위한 CNN 모델

이 프로젝트는 TensorFlow/Keras를 사용하여 고양이와 강아지 이미지를 분류하는 Convolutional Neural Network(CNN)를 구현한 것입니다. 데이터는 Kaggle에서 제공되는 "Dogs vs Cats" 데이터셋을 사용하며, 이미지를 전처리하여 이진 분류 모델을 학습시킵니다.

---

## 목차

1. [데이터셋]
2. [필수 준비사항]
3. [프로젝트 구조]
4. [모델 구조]
5. [훈련 및 검증]
6. [사용 방법]
7. [결과]

---

## 데이터셋

사용된 데이터는 **Kaggle의 Dogs vs Cats 데이터셋**입니다. 데이터 준비 방법:

1. Kaggle에서 데이터셋을 다운로드합니다: Dogs vs Cats Dataset.
2. 데이터를 압축 해제 후 다음과 같은 폴더 구조로 정리합니다:
    
    ```bash
    bash
    코드 복사
    data/
    ├── train/
    │   ├── cats/  # 고양이 이미지
    │   └── dogs/  # 강아지 이미지
    ├── val/
    │   ├── cats/  # 검증용 고양이 이미지
    │   └── dogs/  # 검증용 강아지 이미지
    
    ```
    

---

## 필수 준비사항

코드를 실행하기 전에 다음의 필수 라이브러리를 설치해야 합니다:

- Python (>= 3.8)
- TensorFlow (>= 2.9)
- NumPy
- Matplotlib

라이브러리 설치 명령어:

```bash
bash
코드 복사
pip install tensorflow numpy matplotlib

```

---

## 프로젝트 구조

- `train_ds`: 학습 데이터셋, 이미지 정규화를 포함한 전처리 적용.
- `val_ds`: 검증 데이터셋, 동일한 전처리 적용.
- 모델은 TensorFlow/Keras의 `Sequential` API를 사용해 정의.
- 5 에포크 동안 훈련하며 정확도와 손실을 기록.

---

## 모델 구조

CNN 모델은 다음과 같은 레이어로 구성됩니다:

1. **Conv2D + ReLU**: 이미지에서 특징 추출 (32개의 필터, 3x3 커널).
2. **MaxPooling2D**: 특징 맵 다운샘플링 (2x2 풀 크기).
3. **Conv2D + ReLU**: 더 깊은 특징 추출 (64개의 필터, 3x3 커널).
4. **MaxPooling2D**: 추가 다운샘플링.
5. **Dropout (20%)**: 과적합 방지를 위한 정규화.
6. **Conv2D + ReLU**: 더 깊은 특징 추출 (128개의 필터, 3x3 커널).
7. **MaxPooling2D**: 추가 다운샘플링.
8. **Flatten**: 평탄화하여 Fully Connected Layer로 연결.
9. **Dense + ReLU**: 128개의 뉴런으로 구성된 완전 연결층.
10. **Dropout (20%)**: 추가 정규화.
11. **Dense + Sigmoid**: 이진 분류를 위한 출력층 (1개의 뉴런).

---

## 훈련 및 검증

모델은 다음과 같은 설정으로 컴파일됩니다:

- **손실 함수**: `binary_crossentropy` (이진 분류 문제에 적합).
- **최적화 알고리즘**: Adam.
- **평가지표**: 정확도(accuracy).

훈련은 5 에포크 동안 진행되며, 각 에포크마다 학습 데이터와 검증 데이터의 손실 및 정확도가 기록됩니다.

---

## 사용 방법

1. **데이터 준비**:
    - 데이터셋을 다운로드하고, [데이터셋] 섹션에 설명된 대로 폴더를 구성합니다.
2. **코드 실행**:
아래 코드를 `train_model.py` 파일로 저장하고 실행합니다.
    
    ```python
    python
    코드 복사
    import tensorflow as tf
    
    # 이미지 전처리 함수
    def preprocess_images(image, label):
        image = tf.cast(image / 255.0, tf.float32)
        return image, label
    
    # 데이터 파이프라인 설정
    train_ds = train_ds.map(preprocess_images)
    val_ds = val_ds.map(preprocess_images)
    
    # 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    
    # 모델 컴파일
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    # 모델 훈련
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    
    ```
    
3. **훈련 과정 확인**:
에포크별 훈련 및 검증 정확도와 손실이 기록됩니다.

---

## 결과

5 에포크 훈련 후 모델은 다음과 같은 성능을 보입니다:

- **훈련 정확도**: 약 81.96%
- **검증 정확도**: 약 77.62%
- **검증 손실**: 에포크가 진행됨에 따라 감소.

---

## 향후 과제

- 추가적인 데이터 증강 기법 실험.
- 학습률 및 배치 크기 등의 하이퍼파라미터 튜닝.
- 성능 향상을 위해 더 깊은 모델 구조나 전이 학습(Transfer Learning) 적용.

---
