# Демонстрация функционирования сайта
***
https://drive.google.com/file/d/1ZmomX40hyLHc_1Y70XmKSkbFO4xOJxAE/view?usp=sharing

# Структура нашего проекта: 
- app.py - файл с сайтом
- requirements.txt - файл с необходимыми библиотеками
- result.csv - датасет
- Регрессионный анализ.ipynb - Регрессионный анализ значимости переменных в метриках: удовлетворенности пользователей и качества работы моделей, реализ с помощью логистической модели, которая предназначена для работы с бинарными переменными 

# Описание дашбордов
Самые первые три блока разных цветов отображают три общие метрики по работе чат-бота, в соответствии со значением в ячейке подбирается цвет фона: если точность ответа высокая, то фон будет зеленым, если в районе 60-50%, то оранжевым, если ниже 50 процентов, то красным. Аналогично работает и по другим метрикам.

Под блоками на сайте реализована система фильтрации: кампус, уровень образования, категории вопросов и даты (можно выбрать день, месяц, год или какой-либо кастомный период.

Слева на сайте расположен модуль с фильтрами по показателям технических метрик, в котором можно оценить среднее значение метрики, максимальное и минимальное значение, а также эконометрический анализ метрики, который содержит в себе коэффициент для заданной метрики и стандартную ошибку.

**Spider-chart** позволяет сравнить ответ различных моделей с ответом AI (Ground Truth) решением по четырем техническим метрикам: BERT accuracy, context recall, context precision, literal correctness. 

**Столбчатая диаграмма context recall** позволяет оценить полноту (recall) между ответами Ground Truth и ответами моделей (Giga, Saiga).

**Столбчатая диаграмма context precision** позволяет оценить точность (precision) между ответами Ground Truth и ответами моделей (Giga, Saiga).

**Столбчатая диаграмма correctness literal** позволяет оценить метрику, которая была разобрана в статье (от Леры: https://aclanthology.org/W15-3049.pdf) между Ground Truth решением и моделями (Giga, Saiga). 

**Столбчатая диаграмма BERT (Correctness neural)** позволяет посмотреть смысловую схожесть между ответами моделей (Giga, Saiga) и ответом Ground Truth.

**Model Evaluation** позволяет оценить покрытие запросов пользователя в чат-бот моделями.

**График Stacked Bar Chart** по категориям показывает долю  запросов с уточнениями в разрезе различных категорий вопросов. График позволяет понять, в каких категориях вопросов пользователи чаще требуют дополнительных разъяснений. 

**График матрицы корреляции** показывает взаимосвязь между различными метриками работы чат-бота. Он демонстрирует, как одни показатели влияют на другие. Этот график помогает анализировать, какие параметры чат-бота взаимосвязаны и где можно улучшить его работу.


# Установка проекта
*** 
Склонируйте репозиторий
```bash
git clone https://github.com/voypoka/dashboard_WindBlows.git
cd dashboard_WindBlows
```
Создайте виртуальное окружение
MacOS / Ubuntu
```bash
python3 -m venv venv

source venv/bin/activate
```
Windows
```bash
python -m venv venv

venv/bin/activate 
```

Установите нужные библиотеки
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Запустить приложение
```bash
streamlit run app.py 
```
Видео с установкой: https://drive.google.com/file/d/1ZI38XwWw-7HJDEdhQ4D6FDGChg6v6S0g/view?usp=sharing
