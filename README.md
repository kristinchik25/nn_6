# Практическая работа 6. Введение в PyTorch
# Цели работы
После выполнения рабочей тетради студент должен уметь:

- создавать и преобразовывать тензоры PyTorch;
- выполнять операции над тензорами и понимать broadcasting;
- использовать autograd для вычисления градиентов;
- описывать простую нейронную сеть через torch.nn;
- работать с Dataset и DataLoader;
- запускать базовый цикл обучения и оценивать качество модели;
- адаптировать модель под новую задачу с помощью заморозки слоев.

# Индивидуальное задание, 11 вариант 
| Вариант | Скрытых нейронов | Активация | Оптимизатор | LR    | Batch size | Эпох |
|---------|------------------|-----------|-------------|-------|------------|------|
| 11      | 96               | ReLU      | Adam        | 0.005 | 16         | 6    |


Что сделать:

- загрузить FashionMNIST
- создать модель с одним скрытым слоем указанного размера
- использовать указанную функцию активации, оптимизатор, learning rate, batch_size и количество эпох
- вывести итоговую accuracy на тестовой выборке
- построить/вывести матрицу ошибок или таблицу правильных ответов по классам
- написать вывод: какие параметры дали хороший/плохой результат и почему

# Решение
Код решения 
````
# ========== ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ, ВАРИАНТ 11 ==========
variant = 11

# Параметры из таблицы
hidden_size = 96
activation_name = 'ReLU'
optimizer_name = 'Adam'
learning_rate = 0.005
batch_size = 16
epochs = 6

# 1. Загрузка FashionMNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Определение модели с одним скрытым слоем
class FashionMNIST_OneHidden(nn.Module):
    def __init__(self, hidden_size=96):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.activation = nn.ReLU()  # ReLU
        self.fc2 = nn.Linear(hidden_size, 10)  # 10 классов

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

model = FashionMNIST_OneHidden(hidden_size=hidden_size)

# 3. Определение оптимизатора и функции потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Используется устройство: {device}")

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Эпоха {epoch}/{epochs} | Loss: {avg_loss:.4f}")

# 5. Оценка на тестовой выборке
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\nИтоговая точность (accuracy): {accuracy:.2f}%")

# 6. Матрица ошибок и таблица по классам
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = confusion_matrix(all_labels, all_preds)
# Для наглядности выведем таблицу правильных ответов по классам
print("\nРезультаты по каждому классу:")
for i, cls in enumerate(classes):
    correct = cm[i, i]
    total = np.sum(cm[i, :])
    print(f"{cls:15s}: {correct:4d} / {total:4d}  ({100*correct/total:.1f}%)")

# Построим матрицу ошибок (если нужно)
plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix для FashionMNIST (Вариант 11)")
plt.tight_layout()
plt.show()
````

# Результат

````
Используется устройство: cuda
Эпоха 1/6 | Loss: 0.5378
Эпоха 2/6 | Loss: 0.4573
Эпоха 3/6 | Loss: 0.4364
Эпоха 4/6 | Loss: 0.4251
Эпоха 5/6 | Loss: 0.4154
Эпоха 6/6 | Loss: 0.4080

Итоговая точность (accuracy): 83.99%

Результаты по каждому классу:
T-shirt/top    :  860 / 1000  (86.0%)
Trouser        :  951 / 1000  (95.1%)
Pullover       :  742 / 1000  (74.2%)
Dress          :  848 / 1000  (84.8%)
Coat           :  871 / 1000  (87.1%)
Sandal         :  878 / 1000  (87.8%)
Shirt          :  409 / 1000  (40.9%)
Sneaker        :  968 / 1000  (96.8%)
Bag            :  951 / 1000  (95.1%)
Ankle boot     :  921 / 1000  (92.1%)
````

<img width="539" height="470" alt="image" src="https://github.com/user-attachments/assets/45022ebf-53dd-49d7-a991-4d67cd25d426" />

# Анализ
## Оценка параметров 


| Параметр         | Оценка              | Почему                                                            |
|------------------|---------------------|-------------------------------------------------------------------|
| hidden_size=96   |    хороший         | Достаточная ёмкость без переобучения                             |
| ReLU             |    хороший          | Быстрая сходимость, нет затухания градиента                       |
| Adam             |    хороший          | Адаптивный, подходит для этой задачи                              |
| LR=0.005         |    хороший          | Стабильное обучение                                               |
| batch_size=16    |    хороший          | Шум градиентов помогает обобщению                                 |
| epochs=6         |     недостаточно     | Точность растёт, стоит увеличить до 15–20                         |
| 1 скрытый слой   | недостаточно      | Сложные классы требуют большей глубины                            |

## Наблюдения
- **Лучше всего распознаются классы:** Sneaker (96.8%), Trouser (95.1%), Bag (95.1%), Ankle boot (92.1%)

- **Хуже всего распознаются классы:** Shirt (40.9%), Pullover (74.2%), Dress (84.8%)
## Выводы
Модель с одним скрытым слоем (96 нейронов) и активацией ReLU показала удовлетворительное качество классификации FashionMNIST (83.99%). Лучше всего распознаются классы с характерными устойчивыми признаками: обувь (Sneaker, Ankle boot), брюки (Trouser) и сумки (Bag). Хуже всего классифицируется класс Shirt (40.9%), что объясняется высокой визуальной схожестью футболок с другими предметами одежды верхней части тела (Pullover, Dress, Coat). Матрица ошибок подтверждает систематическую путаницу между этими классами. Для улучшения результатов необходимо увеличить количество эпох обучения до 20 и  добавить минимум один скрытый слой 
