# Импорт необходимых библиотек
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загрузка набора данных о вине
wine = datasets.load_wine()

# Вывод атрибутов набора данных
print("Атрибуты набора данных:", dir(wine))

# Преобразование данных в массивы для обучения
X = wine.data
y = wine.target

# Нормализация данных для улучшения работы модели
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки (95% на обучение, 5% на тестирование)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=42)

# Инициализация и обучение MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,50),  # Увеличение количества нейронов в скрытом слое
                    activation='relu',          # Использование ReLU для лучшей производительности
                    alpha=1e-5,                # Меньший коэффициент регуляризации
                    solver='adam',             # Использование Adam для оптимизации
                    max_iter=1000,             # Увеличение числа итераций для обучения
                    random_state=1,
                    verbose=False)             # Отключение вывода логов

# Обучение модели
mlp.fit(X_train, y_train)

# Вывод значений потерь на каждой итерации
print("Значения потерь на каждой итерации:")
for i, loss in enumerate(mlp.loss_curve_):
    print(f"Итерация {i + 1} = {loss:.4f}")

# Построение графика потерь во время обучения
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, 'o-')
plt.title('График потерь во время обучения')
plt.xlabel('Количество итераций')
plt.ylabel('Потери')
plt.grid()
plt.show()

# Функция для отображения матрицы тестовых данных в виде тепловой карты
def display_test_data_heatmap(X_test):
    plt.figure(figsize=(12, 8))
    sns.heatmap(X_test, annot=True, fmt=".2f", cmap='viridis', xticklabels=wine.feature_names)
    plt.title('Матрица тестовых данных')
    plt.xlabel('Характеристики')
    plt.ylabel('Примеры')
    plt.show()

# Оценка точности модели на тестовом наборе
predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Точность на тестовом наборе: {accuracy * 100:.2f}%')

# Вызов функции для отображения матрицы тестовых данных в отдельном окне
display_test_data_heatmap(X_test)
