import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Настройка страницы и тёмного фона
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Тёмный фон страницы */
    body, .main {
        background-color: #000000 !important;
        color: #FFFFFF;
    }
    .top-panel {
    display: flex;          /* Располагаем карточки в ряд */
    gap: 20px;              /* Отступы между карточками */
    margin-bottom: 20px;    /* Отступ снизу */
    flex-wrap: wrap;        /* При нехватке места карточки переносятся на новую строку */
}

.card {
    border-radius: 10px;
    padding: 20px;
    flex: 1;
    min-width: 200px;
    color: #FFFFFF;
    text-align: center;
}

.card.accuracy { background-color: #50D620; }   /* Светло-зелёный */
.card.speed { background-color: #FB851D; }      /* Жёлтый */
.card.likes { background-color: #F90000; }      /* Красный */

.card h3 {
    margin: 0;
    font-size: 1.1rem;
    color: #FFFFFF;
}

.card .value {
    font-size: 2rem;
    font-weight: bold;
    margin-top: 10px;
}
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Загрузка данных
df = pd.read_csv("result.csv", encoding="utf-8")

st.title("Дашборд: мониторинг качества чат-бота (Итоговый датасет)")

with st.expander("ℹ️ Отладочная информация"):
    st.write("Список колонок:", df.columns.tolist())
    st.dataframe(df.head())

st.markdown("<div class='top-panel'>", unsafe_allow_html=True)

#Подсчёт метрик

#Средняя точность
bert_columns = [
    "answer_correctness_neural - Ответ AI и Giga",
    "answer_correctness_neural - Ответ AI и Saiga",
    "answer_correctness_neural - Saiga и Giga"
]

# Вычисляем среднее значение по этим столбцам и умножаем на 100 для процентов
avg_accuracy = df[bert_columns].mean().mean() * 100

#Среднее время ответа
if "Время ответа модели (сек)" not in df.columns:
    print("В датасете нет столбца 'Время ответа'. Проверьте название столбца.")
else:
    avg_speed = df["Время ответа модели (сек)"].mean()


if "random_flag" not in df.columns:
    print("В датасете нет столбца 'random_flag'. Проверьте название столбца.")
else:
    likes_percentage = df["random_flag"].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    # Карточка 1: Средняя точность
    st.markdown(f"""
    <div class='card accuracy'>
        <h3>Средняя точность</h3>
        <div class='value'>{avg_accuracy:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Карточка 2: Средняя скорость ответов
    st.markdown(f"""
    <div class='card speed'>
        <h3>Средняя скорость ответов</h3>
        <div class='value'>{avg_speed:.2f} сек</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Карточка 3: Процент лайков
    st.markdown(f"""
    <div class='card likes'>
        <h3>Процент лайков</h3>
        <div class='value'>{likes_percentage:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# 3. Списки для фильтров и метрик
campuses = ["Москва", "Нижний Новгород", "Санкт-Петербург", "Пермь"]
education_levels = ["Бакалавриат", "Магистратура", "Специалитет", "Аспирантура"]
question_categories = [
    "Учеба", "Финансовые вопросы", "Цифровые сервисы и Техподдержка",
    "Обратная связь (в т.ч СОП)", "Социальные вопросы", "Наука",
    "Военка", "Внеучебка", "Другое",
]

standard_metrics = [
    "context_recall - AI и Saiga",
    "context_recall - Ответ AI и Giga",
    "context_recall - Saiga и Giga",
    "context_precision - Ответ AI и Saiga",
    "context_precision - Ответ AI и Giga",
    "context_precision - Saiga и Giga",
    "chrF - Ответ AI и Saiga",
    "chrF - Ответ AI и Giga",
    "chrF - Saiga и Giga",
    "answer_correctness_neural - Ответ AI и Giga",
    "answer_correctness_neural - Ответ AI и Saiga",
    "answer_correctness_neural - Saiga и Giga"
]

df["random_timestamp"] = pd.to_datetime(df["random_timestamp"])

# 4. Блок с фильтрами (общий)
st.subheader("Фильтры")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_campus = st.multiselect("Кампус", campuses, default=[])
    if not selected_campus:
        selected_campus = campuses

with col2:
    selected_edu = st.multiselect("Уровень образования", education_levels, default=[])
    if not selected_edu:
        selected_edu = education_levels

with col3:
    selected_cat = st.multiselect("Категория вопроса", question_categories, default=[])
    if not selected_cat:
        selected_cat = question_categories

with col4:
    selected_dates = st.date_input(
        "Выберите временной диапазон",
        value=[df["random_timestamp"].min().date(), df["random_timestamp"].max().date()]
    )

if isinstance(selected_dates, tuple):
    selected_dates = list(selected_dates)

if isinstance(selected_dates, list):
    if len(selected_dates) == 1:
        start_date = selected_dates[0]
        end_date = selected_dates[0]
    elif len(selected_dates) >= 2:
        start_date, end_date = selected_dates[0], selected_dates[1]
else:
    start_date = selected_dates
    end_date = selected_dates
# Применяем фильтры
df_filtered = df[
    (df["Кампус"].isin(selected_campus)) &
    (df["Уровень образования"].isin(selected_edu)) &
    (df["Категория вопроса"].isin(selected_cat)) &
    (df["random_timestamp"].dt.date >= start_date) &
    (df["random_timestamp"].dt.date <= end_date)
]

st.markdown("---")

# 5. Разделяем страницу на три колонки
col1, col2, col3 = st.columns(3)

# =========================
#  Функции для расчётов
# =========================
def choose_metric():
    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = standard_metrics[0]
    st.session_state["selected_metric"] = st.selectbox("Выберите метрику:", standard_metrics)

def show_mean(df_filtered):
    """
    Выводит среднее значение выбранной метрики в отдельном модуле.
    """
    if "selected_metric" not in st.session_state:
        st.write("Сначала выберите метрику.")
        return
    metric = st.session_state["selected_metric"]
    if metric not in df_filtered.columns:
        st.write(f"Столбец '{metric}' не найден в данных.")
    else:
        avg_val = df_filtered[metric].mean()
        st.markdown(f"""
        <div style="background-color: #242630; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
          <h3 style="color: white;">Среднее значение выбранной метрики</h3>
          <p style="color: white; font-size: 1.5rem;">{avg_val:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def show_minmax(df_filtered):
    """
    Выводит максимальное и минимальное значение выбранной метрики в отдельном модуле.
    """
    if "selected_metric" not in st.session_state:
        st.write("Сначала выберите метрику.")
        return
    metric = st.session_state["selected_metric"]
    if metric not in df_filtered.columns:
        st.write(f"Столбец '{metric}' не найден в данных.")
    else:
        max_val = df_filtered[metric].max()
        min_val = df_filtered[metric].min()
        st.markdown(f"""
        <div style="background-color: #242630; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
          <h3 style="color: white;">Максимальное и минимальное значение</h3>
          <p style="color: white; font-size: 1.5rem;">Максимум: {max_val:.3f} | Минимум: {min_val:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def logistic_analysis(df_filtered):
    """
    Выводит результаты эконометрического анализа (логистическая регрессия)
    для выбранной метрики в отдельном модуле.
    """
    if "selected_metric" not in st.session_state:
        st.write("Сначала выберите метрику.")
        return
    metric = st.session_state["selected_metric"]

    if "random_flag" not in df_filtered.columns:
        st.write("Нет столбца 'random_flag' для эконометрического анализа.")
        return

    missing_cols = [col for col in standard_metrics if col not in df_filtered.columns]
    if missing_cols:
        st.write("Отсутствуют столбцы для анализа:", missing_cols)
        return

    y = df_filtered["random_flag"]
    X = df_filtered[standard_metrics]
    if len(X) == 0:
        st.write("Нет данных для построения модели (после фильтрации).")
        return

    X_const = sm.add_constant(X, has_constant="add")
    if len(np.unique(y)) < 2:
        st.write("В отфильтрованных данных только один класс 'random_flag'. Нельзя обучить логистическую регрессию.")
        return

    try:
        model = sm.Logit(y, X_const).fit(disp=0)
    except Exception as e:
        st.write("Ошибка при построении модели:", e)
        return

    if metric not in model.params.index:
        st.write(f"Метрика '{metric}' не найдена в построенной модели.")
        return

    coef = model.params[metric]
    se = model.bse[metric]
    st.markdown(f"""
    <div style="background-color: #242630; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
      <h3 style="color: white;">Эконометрический анализ</h3>
      <p style="color: white; font-size: 1.5rem;">Коэффициент для '{metric}': {coef:.3f}</p>
      <p style="color: white; font-size: 1.5rem;">Стандартная ошибка: {se:.3f}</p>
    </div>
    """, unsafe_allow_html=True)


def show_covariance_matrix(df_filtered):
    """
    Вычисляет и отображает корреляционную (ковариационную) матрицу для random_flag и стандартных метрик.
    """
    if "random_flag" not in df_filtered.columns:
        st.write("Нет столбца 'random_flag' для построения матрицы.")
        return

    needed_metrics = [
        "context_recall - AI и Saiga",
        "context_recall - Ответ AI и Giga",
        "context_recall - Saiga и Giga",
        "context_precision - Ответ AI и Saiga",
        "context_precision - Ответ AI и Giga",
        "context_precision - Saiga и Giga",
        "chrF - Ответ AI и Saiga",
        "chrF - Ответ AI и Giga",
        "chrF - Saiga и Giga",
        "answer_correctness_neural - Ответ AI и Giga",
        "answer_correctness_neural - Ответ AI и Saiga",
        "answer_correctness_neural - Saiga и Giga"
    ]
    missing_cols = [col for col in needed_metrics if col not in df_filtered.columns]
    if missing_cols:
        st.write("Отсутствуют столбцы для построения матрицы:", missing_cols)
        return

    y = df_filtered["random_flag"]
    X = df_filtered[needed_metrics]
    data_matrix = pd.concat([y, X], axis=1)
    corr_matrix = data_matrix.corr()

    # Фигура с фоном #242630
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#242630")
    ax.set_facecolor("#242630")
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"color": "white"},
        cbar_kws={"label": "Correlation", "ticks": [-1, 0, 1]}
    )
    ax.set_title("Корреляционная матрица", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlation", color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    st.pyplot(fig)

def show_radar_chart(df_filtered, categories=None):
    """
    Строит spider-chart (радарную диаграмму) для сравнения метрик по парам моделей.
    """
    if categories:
        data_for_chart = df_filtered[df_filtered['Категория вопроса'].isin(categories)]
    else:
        data_for_chart = df_filtered

    metrics_order = ['context_recall', 'context_precision', 'answer_correctness_literal', 'answer_correctness_neural']
    comparison_names = ['AI - Saiga', 'AI - Giga', 'Saiga - Giga']

    comparison_columns = {
        'AI - Saiga': {
            'context_recall': 'context_recall - AI и Saiga',
            'context_precision': 'context_precision - Ответ AI и Saiga',
            'answer_correctness_literal': 'chrF - Ответ AI и Saiga',
            'answer_correctness_neural': 'answer_correctness_neural - Ответ AI и Saiga'
        },
        'AI - Giga': {
            'context_recall': 'context_recall - Ответ AI и Giga',
            'context_precision': 'context_precision - Ответ AI и Giga',
            'answer_correctness_literal': 'chrF - Ответ AI и Giga',
            'answer_correctness_neural': 'answer_correctness_neural - Ответ AI и Giga'
        },
        'Saiga - Giga': {
            'context_recall': 'context_recall - Saiga и Giga',
            'context_precision': 'context_precision - Saiga и Giga',
            'answer_correctness_literal': 'chrF - Saiga и Giga',
            'answer_correctness_neural': 'answer_correctness_neural - Saiga и Giga'
        }
    }

    data = {}
    for comp in comparison_names:
        comp_data = []
        for metric in metrics_order:
            col = comparison_columns[comp][metric]
            if col.startswith("chrF"):
                value = data_for_chart[col].mean()
            else:
                value = data_for_chart[col].mean() * 100
            comp_data.append(value)
        data[comp] = comp_data

    labels = np.array(['Context Recall', 'Context Precision', 'Literal Correctness', 'BERT'])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True}, facecolor="#242630")
    ax.set_facecolor("#242630")
    colors = sns.color_palette('Set2', 3)

    for idx, (comp, values) in enumerate(data.items()):
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=comp)
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    fig.patch.set_facecolor('#242630')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color="white")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
    plt.ylim(0, 100)
    plt.title('Сравнение метрик по парам моделей', size=15, pad=40, color="white")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)

def model_eval(df_filtered):
    # Создаем копию датафрейма и группируем значения
    df_modified = df_filtered.copy()
    df_modified['Кто лучше?'] = df_modified['Кто лучше?'].replace(regex=r'.*Вопрос не по теме.*', value='Вопрос не по теме')
    counts = df_modified['Кто лучше?'].value_counts()

    colors = sns.color_palette('pastel')
    explode = [0.05] * len(counts)

    # Создаем фигуру с заданным фоном
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#242630")
    ax.set_facecolor("#242630")

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=140,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        textprops={'fontsize': 12, 'color': 'white'}
    )

    ax.set_title('Распределение ответов: Кто лучше?', fontsize=16, fontweight='bold', color="white")
    plt.tight_layout()
    st.pyplot(fig)


def show_stacked_bar_by_category_seaborn(df_filtered, categories=None):
    # Если заданы дополнительные категории, отфильтруем df_filtered
    if categories:
        df_local = df_filtered[df_filtered['Категория вопроса'].isin(categories)].copy()
    else:
        df_local = df_filtered.copy()

    # Создаем вспомогательный столбец "Уточнения"
    df_local['Уточнения'] = df_local['Ответ AI (уточнение)'].notnull().map({
        True: 'С уточнениями',
        False: 'Без уточнений'
    })

    # Создаем фигуру с фоном #242630
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#242630")
    ax.set_facecolor("#242630")

    # Строим горизонтальную стековую диаграмму (multiple='fill' нормализует столбцы)
    sns.histplot(
        data=df_local,
        y='Категория вопроса',  # категории по оси Y
        hue='Уточнения',  # разбивка по наличию уточнений
        multiple='fill',  # нормированные значения (0..1)
        discrete=True,
        alpha=0.6,
        shrink=0.8,
        ax=ax
    )

    ax.set_xlabel('Доля запросов (%)', color="white")
    ax.set_ylabel('Категория вопроса', color="white")
    ax.set_title('Процент запросов с уточнениями по категориям', color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    st.pyplot(fig)


def show_metrics(df_filtered, metric_name):
    """
    Функция для построения графика с автоматической проверкой необходимости масштабирования.
    Использует уже отфильтрованные данные (df_filtered).
    """
    no_scale_metrics = ["chrF"]  # Метрики, которые уже в диапазоне 0-100
    df_local = df_filtered.copy()
    metric_columns = [col for col in df_local.columns if metric_name in col]
    if not metric_columns:
        st.write("Нет данных для выбранной метрики.")
        return

    df_metrics = df_local[metric_columns].copy()
    if metric_name not in no_scale_metrics:
        df_metrics = df_metrics * 100  # Преобразуем в проценты

    df_metrics["Категория вопроса"] = df_local["Категория вопроса"].values
    df_metrics.reset_index(drop=True, inplace=True)

    df_melted = df_metrics.melt(
        id_vars="Категория вопроса",
        var_name="Модели",
        value_name="Значение"
    )
    df_grouped = df_melted.groupby(["Категория вопроса", "Модели"], as_index=False).mean()

    fig = plt.figure(figsize=(12, 6), facecolor="#242630")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#242630")

    sns.barplot(
        data=df_grouped,
        x="Категория вопроса",
        y="Значение",
        hue="Модели",
        palette="Set2",
        ax=ax
    )

    if metric_name == "answer_correctness_neural":
        metric_title = "BERT"
    else:
        metric_title = metric_name

    ax.set_title(f"{metric_title} по категориям", color="white")
    ax.set_ylim(0, 100)
    ylabel = "Значение (%)" if metric_name not in no_scale_metrics else "Значение"
    ax.set_ylabel(ylabel, color="white")
    ax.set_xlabel("Категория вопроса", color="white")
    ax.tick_params(axis="x", rotation=45, colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    st.pyplot(fig)

# =========================
#  Функция для нового графика в третьем столбце
# =========================
def show_context_recall(df_filtered):
    """
    График по context_recall (AI - Saiga, AI - Giga, Saiga - Giga) в стиле #242630.
    Использует глобально отфильтрованные данные (df_filtered).
    """
    # Извлекаем нужные колонки и масштабируем
    df_need_columns = df_filtered[[
        "context_recall - AI и Saiga",
        "context_recall - Ответ AI и Giga",
        "context_recall - Saiga и Giga"
    ]] * 100

    mean_values = df_need_columns.mean()
    df_mean = pd.DataFrame({
        'Категории': ['AI - Saiga', 'AI - Giga', 'Saiga - Giga'],
        'Схожесть, %avg': mean_values.values
    })

    # Создаем фигуру с фоном #242630
    fig = plt.figure(figsize=(6, 4), facecolor="#242630")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#242630")

    sns.barplot(
        data=df_mean,
        x='Категории',
        y='Схожесть, %avg',
        palette='Set2',
        dodge=False,
        ax=ax
    )

    ax.set_title('Context recall', color="white")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Схожесть (%)", color="white")

    ax.tick_params(axis='x', colors="white")
    ax.tick_params(axis='y', colors="white")

    plt.tight_layout()
    st.pyplot(fig)

def show_context_precision(df_filtered):
    """
    График по context_precision (AI - Saiga, AI - Giga, Saiga - Giga) в стиле #242630.
    Использует глобально отфильтрованные данные (df_filtered).
    """
    # Извлекаем нужные колонки и масштабируем (умножаем на 100)
    df_need_columns = df_filtered[[
        "context_precision - Ответ AI и Saiga",
        "context_precision - Ответ AI и Giga",
        "context_precision - Saiga и Giga"
    ]] * 100

    # Вычисляем средние значения для каждой категории
    mean_values = df_need_columns.mean()
    df_mean = pd.DataFrame({
        'Категории': ['AI - Saiga', 'AI - Giga', 'Saiga - Giga'],
        'Схожесть, %avg': mean_values.values
    })

    # Создаем фигуру с фоном #242630
    fig = plt.figure(figsize=(6, 4), facecolor="#242630")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#242630")

    # Строим барплот
    sns.barplot(
        data=df_mean,
        x='Категории',
        y='Схожесть, %avg',
        palette='Set2',
        dodge=False,
        ax=ax
    )

    ax.set_title('Context precision', color="white")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")  # При необходимости можно добавить подпись
    ax.set_ylabel("Схожесть (%)", color="white")
    ax.tick_params(axis='x', colors="white")
    ax.tick_params(axis='y', colors="white")
    plt.tight_layout()
    st.pyplot(fig)


def show_answer_correctness_literal(df_filtered):
    """
    Строит график для 'Correctness literal' (AI - Saiga, AI - Giga, Saiga - Giga)
    с оформлением: фон #242630, белые подписи и т.д.
    Использует глобально отфильтрованные данные (df_filtered).
    """
    # Извлекаем нужные колонки (без масштабирования, т.к. данные уже в нужном диапазоне)
    df_need_columns = df_filtered[[
        "chrF - Ответ AI и Saiga",
        "chrF - Ответ AI и Giga",
        "chrF - Saiga и Giga"
    ]]

    # Вычисляем средние значения для каждой категории
    mean_values = df_need_columns.mean()
    df_mean = pd.DataFrame({
        'Категории': ['AI - Saiga', 'AI - Giga', 'Saiga - Giga'],
        'Схожесть, %avg': mean_values.values
    })

    # Создаем фигуру с фоном #242630
    fig = plt.figure(figsize=(6, 4), facecolor="#242630")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#242630")

    # Строим barplot
    sns.barplot(
        data=df_mean,
        x='Категории',
        y='Схожесть, %avg',
        palette='Set2',
        dodge=False,
        ax=ax
    )

    ax.set_title('Correctness literal', color="white")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Схожесть (%)", color="white")
    ax.tick_params(axis='x', colors="white")
    ax.tick_params(axis='y', colors="white")
    plt.tight_layout()
    st.pyplot(fig)

def show_answer_correctness_neural(df_filtered):
    """
    Строит график для 'BERT' (AI - Saiga, AI - Giga, Saiga - Giga)
    с оформлением: фон #242630, белые подписи и т.д.
    Использует глобально отфильтрованные данные (df_filtered).
    """
    # Извлекаем нужные колонки и масштабируем (умножаем на 100)
    df_need_columns = df_filtered[[
        "answer_correctness_neural - Ответ AI и Saiga",
        "answer_correctness_neural - Ответ AI и Giga",
        "answer_correctness_neural - Saiga и Giga"
    ]] * 100

    # Вычисляем средние значения для каждой категории
    mean_values = df_need_columns.mean()
    df_mean = pd.DataFrame({
        'Категории': ['AI - Saiga', 'AI - Giga', 'Saiga - Giga'],
        'Схожесть, %avg': mean_values.values
    })

    # Создаем фигуру с фоном #242630
    fig = plt.figure(figsize=(6, 4), facecolor="#242630")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#242630")

    # Строим barplot
    sns.barplot(
        data=df_mean,
        x='Категории',
        y='Схожесть, %avg',
        palette='Set2',
        dodge=False,
        ax=ax
    )

    ax.set_title('BERT', color="white")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Схожесть (%)", color="white")
    ax.tick_params(axis='x', colors="white")
    ax.tick_params(axis='y', colors="white")
    plt.tight_layout()
    st.pyplot(fig)


# =========================
#  Колонка 1
# =========================
with col1:
    st.subheader("Выбор стандартной метрики")
    choose_metric()

    st.markdown("---")

    #st.subheader("Среднее значение выбранной метрики")
    show_mean(df_filtered)

    st.markdown("---")

    #st.subheader("Максимальное и минимальное значение")
    show_minmax(df_filtered)

    st.markdown("---")

    #st.subheader("Эконометрический анализ")
    logistic_analysis(df_filtered)

    st.markdown("---")


# =========================
#  Колонка 2
# =========================
with col2:
    st.subheader("Spider-chart")
    show_radar_chart(df_filtered)

    st.markdown("---")

    st.subheader("Model Evaluation")
    model_eval(df_filtered)

    st.markdown("---")

    st.subheader("Stacked Bar Chart по категориям")
    show_stacked_bar_by_category_seaborn(df_filtered)

    st.markdown("---")


# =========================
#  Колонка 3: Новый график
# =========================
with col3:
    st.subheader("Context Recall (обобщённый)")
    show_context_recall(df_filtered)

    st.markdown("---")

    st.subheader("Context Precision (обобщённый)")
    show_context_precision(df_filtered)

    st.markdown("---")

    st.subheader("Correctness literal (обобщённый)")
    show_answer_correctness_literal(df_filtered)

    st.markdown("---")

    st.subheader("BERT (Correctness neural)")
    show_answer_correctness_neural(df_filtered)

# =========================
#  Корреляционная матрица (вне столбцов)
# =========================
st.subheader("Корреляционная матрица")
show_covariance_matrix(df_filtered)

st.markdown("---")

st.subheader("График по категориям")
local_metric = st.selectbox(
    "Выберите метрику для графика:",
    options=["context_recall", "context_precision", "chrF", "answer_correctness_neural"],
    key="local_metric"
)
show_metrics(df_filtered, local_metric)
# =========================
#  Итоговая информация
# =========================
st.markdown("---")
st.write(f"Найдено строк после фильтрации: {len(df_filtered)}")

with st.expander("Посмотреть отфильтрованные данные"):
    st.dataframe(df_filtered)
