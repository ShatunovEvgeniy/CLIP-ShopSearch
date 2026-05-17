# Fashion CLIP Search System

Система семантического поиска товаров одежды по текстовым запросам на базе дообученной модели **CLIP** (Contrastive Language-Image Pre-training).

## Задача

Реализовать поисковую систему, которая позволяет пользователю находить товары в каталоге интернет-магазина, описывая их естественным языком. 

**Примеры запросов:**
- `"red summer dress with floral pattern"`
- `"black leather jacket"`
- `"mickey mouse t-shirt"`
- `"elegant evening gown"`

Система вычисляет векторные представления (эмбеддинги) изображений и текстов в едином пространстве и находит наиболее релевантные товары по косинусной близости.

## Архитектура

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Текстовый      │     │  CLIP Model     │     │  Векторная      │
│  запрос         │────▶│  (fine-tuned)   │────▶│  база данных   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Визуализация   │◀────│  Ранжирование   │◀────│  Косинусная     │
│  результатов    │     │  по схожести    │     │  схожесть       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Компоненты

| Компонент | Описание |
|-----------|----------|
| `FashionCLIPDataset` | Кастомный DataLoader для пар изображение-текст |
| `extract_features()` | Универсальное извлечение тензоров из вывода CLIP |
| `train_epoch()` / `validate()` | Циклы обучения с логированием CLIP-score |
| `FashionSearchSystem` | Класс поисковой системы с методами `search()` и `visualize_results()` |
| `compute_image_embeddings()` | Предвычисление эмбеддингов для ускорения поиска |

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка данных
```python
import kagglehub
path = kagglehub.dataset_download("nirmalsankalana/fashion-product-text-images-dataset")
```

### 3. Загрузка артефактов
Скачайте предобученные веса и эмбеддинги:  
🔗 [https://disk.yandex.ru/d/OtxdkAxfwtdiMg](https://disk.yandex.ru/d/OtxdkAxfwtdiMg)

```python
# Загрузка модели
checkpoint = torch.load('checkpoints/clip_epoch_3.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Загрузка эмбеддингов
image_embeddings = torch.load('image_embeddings.pt')
```

## Метрики

| Метрика | Значение |
|---------|----------|
| Базовая модель | `openai/clip-vit-base-patch32` |
| Размер эмбеддинга | 512 |
| Разрешение изображений | 224×224 |
| Целевой CLIP-score | > 0.30 |
| Время поиска (1 запрос) | ~15–30 мс* |

*\*После предвычисления эмбеддингов изображений*

## Структура репозитория

```
├── data/
│   ├── data.csv                 # Метаданные товаров
│   └── *.jpg                    # Изображения
├── checkpoints/
│   └── clip_epoch_*.pt          # Чекпоинты модели
├── image_embeddings.pt          # Предвычисленные эмбеддинги [N×512]
├── fashion_search_system.pkl    # Сериализованная система поиска
├── training_curves.png          # Графики обучения
├── main.py                      # Точка входа
└── README.md
```

## Источники данных

- **Датасет**: [Fashion Product Text-Images Dataset](https://www.kaggle.com/datasets/nirmalsankalana/fashion-product-text-images-dataset)  
- **Веса модели и эмбеддинги**: [Yandex Disk](https://disk.yandex.ru/d/OtxdkAxfwtdiMg)
