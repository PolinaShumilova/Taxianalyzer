import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class PriceOptimizer:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.feature_importance = None
        self.features = []
        self.modeling_df = None
        self.success_df = None
        self.clean_and_prepare_data()

    def clean_and_prepare_data(self):
        df = self.df.copy()

        #Удаление строк с пропущенными ключевыми ценами
        if 'price_bid_local' in df.columns:
            df = df.dropna(subset=['price_bid_local'])
        elif 'price_start_local' in df.columns:
            df = df.dropna(subset=['price_start_local'])

        #Целевая переменная
        if 'price_bid_local' in df.columns:
            df['successful_price'] = df['price_bid_local']
        elif 'price_start_local' in df.columns:
            df['successful_price'] = df['price_start_local']
        else:
            df['successful_price'] = np.nan

        features = []

        #Базовые числовые признаки
        if 'distance_in_meters' in df.columns:
            df['distance_km'] = df['distance_in_meters'] / 1000
            features.append('distance_km')

        if 'duration_in_seconds' in df.columns:
            df['duration_minutes'] = df['duration_in_seconds'] / 60
            features.append('duration_minutes')

        if 'pickup_in_seconds' in df.columns:
            df['pickup_minutes'] = df['pickup_in_seconds'] / 60
            features.append('pickup_minutes')

        #Временные признаки
        if 'order_timestamp' in df.columns:
            df['order_timestamp'] = pd.to_datetime(df['order_timestamp'], errors='coerce')
            df = df.dropna(subset=['order_timestamp'])
            df['hour'] = df['order_timestamp'].dt.hour
            df['day_of_week'] = df['order_timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) |
                                  ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
            features.extend(['hour', 'day_of_week', 'is_weekend', 'is_rush_hour'])

        #Рейтинг водителя
        if 'driver_rating' in df.columns:
            df['driver_rating'] = df['driver_rating'].fillna(df['driver_rating'].median())
            features.append('driver_rating')

        #Класс машины
        if 'carname' in df.columns:
            df['car_class'] = df['carname'].map(self._get_car_class)
            features.append('car_class')

        #Платформа
        if 'platform' in df.columns:
            platform_dummies = pd.get_dummies(df['platform'].fillna('unknown'), prefix='platform')
            df = pd.concat([df, platform_dummies], axis=1)
            platform_features = [c for c in platform_dummies.columns.tolist()]
            features.extend(platform_features)

        #Признаки спроса
        if 'order_timestamp' in df.columns and 'price_bid_local' in df.columns:
            df = self._add_demand_features(df)
            if 'orders_last_hour' in df.columns:
                features.append('orders_last_hour') #Количество заказов за последний час
            if 'avg_price_last_hour' in df.columns:
                features.append('avg_price_last_hour') #Средняя цена за последний час

        #Успешные заказы
        if 'is_done' in df.columns:
            success_mask = df['is_done'].astype(str).str.lower().isin(['done', 'true', '1', 'yes'])
            success_df = df[success_mask].copy()
        else:
            success_df = df.copy()

        self.success_df = success_df
        self.features = features

        #Создание датасет для моделирования
        modeling_cols = [c for c in features if c in success_df.columns] + ['successful_price']
        self.modeling_df = success_df[modeling_cols].dropna()

        #Преобразование признаков в числовой формат
        for c in self.features:
            if c in self.modeling_df.columns:
                if self.modeling_df[c].dtype == 'object':
                    try:
                        self.modeling_df[c] = pd.to_numeric(self.modeling_df[c], errors='coerce')
                    except Exception:
                        pass

        #Удаление строк
        self.modeling_df = self.modeling_df.dropna()

    def _get_car_class(self, carname):
        premium_cars = ['Volkswagen', 'Skoda', 'Hyundai', 'Kia', 'Ford', 'Mazda']
        comfort_cars = ['Hyundai', 'Kia', 'Ford', 'Mazda', 'Toyota', 'Honda']
        economy_cars = ['LADA', 'ВАЗ', 'Лада', 'Гранта', 'Калина', 'Веста']
        name = str(carname)
        if any(prem.lower() in name for prem in premium_cars):
            return 3  # Премиум
        if any(comfort.lower() in name for comfort in comfort_cars):
            return 2  # Комфорт
        if any(econ.lower() in name for econ in economy_cars):
            return 1  # Эконом
        return 1

    def _add_demand_features(self, df):
        df = df.sort_values('order_timestamp').reset_index(drop=True)
        #Количество заказов и средняя цена за последний час
        try:
            df['orders_last_hour'] = df.rolling('1H', on='order_timestamp')['order_timestamp'].count()
            df['avg_price_last_hour'] = df.rolling('1H', on='order_timestamp')['price_bid_local'].mean()
        except Exception:
            df['orders_last_hour'] = df['order_timestamp'].dt.floor('H').map(df.groupby(df['order_timestamp'].dt.floor('H'))['order_timestamp'].count())
            df['avg_price_last_hour'] = df['order_timestamp'].dt.floor('H').map(df.groupby(df['order_timestamp'].dt.floor('H'))['price_bid_local'].mean())

        df['orders_last_hour'] = df['orders_last_hour'].fillna(1)
        df['avg_price_last_hour'] = df['avg_price_last_hour'].fillna(df['price_bid_local'].mean())
        return df

    def train_model(self):
        if self.modeling_df is None or len(self.modeling_df) < 10:
            available = len(self.modeling_df) if self.modeling_df is not None else 0
            return False, f"Недостаточно данных для обучения модели (требуется >=10 записей, доступно: {available})."

        X = self.modeling_df[self.features]
        y = self.modeling_df['successful_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.feature_importance = pd.DataFrame({'feature': self.features, 'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)

        return True, (mae, r2)

    def predict_optimal_price(self, input_features):
        if self.model is None:
            raise ValueError("Модель не обучена")

        input_df = pd.DataFrame([input_features])

        for feature in self.features:
            if feature not in input_df.columns:
                if feature in ['orders_last_hour', 'avg_price_last_hour'] and feature in self.modeling_df.columns:
                    input_df[feature] = self.modeling_df[feature].mean()
                else:
                    input_df[feature] = 0

        #Приведение к порядку колонок модели
        X = input_df[self.features]

        pred = float(self.model.predict(X)[0])
        pred = self._apply_business_rules(pred, input_features)
        min_price = input_features.get('min_price', 150)
        return max(pred, min_price)

    def _apply_business_rules(self, predicted_price, input_features):
        price = predicted_price

        if 'duration_minutes' in input_features:
            duration = input_features['duration_minutes']
            if duration > 90:  # более 1.5 часов +20%
                price *= 1.2
            elif duration > 60:  # 1-1.5 часа +15%
                price *= 1.15
            elif duration > 45:  # 45-60 минут +10%
                price *= 1.1
            elif duration > 30:  # 30-45 минут +5%
                price *= 1.05
            elif duration < 10:  # менее 10 минут -8%
                price *= 0.92
            elif duration < 15:  # 10-15 минут -5%
                price *= 0.95

        if 'distance_km' in input_features:
            distance = input_features['distance_km']
            if distance > 30:
                price *= 1.25  # +25% за очень длинные (>30 км)
            elif distance > 20:
                price *= 1.15  # +15% за длинные (20-30 км)
            elif distance > 15:
                price *= 1.12  # +12% за средние-длинные (15-20 км)
            elif distance > 10:
                price *= 1.05  # +5% за средние (10-15 км)
            elif distance < 3:
                price *= 0.88  # -12% за очень короткие (<3 км)
            elif distance < 5:
                price *= 0.92  # -8% за короткие (3-5 км)

        if 'hour' in input_features:
            hour = input_features['hour']
            if hour in [7, 8, 9, 17, 18, 19]:
                price *= 1.15
            elif hour in [0, 1, 2, 3, 4, 5]:
                price *= 1.2

        if 'day_of_week' in input_features:
            day = input_features['day_of_week']
            #Выходные - повышенный спрос, +15%
            if day in [5, 6]:
                price *= 1.15
            #Понедельник, пятница - высокий спрос, +10%
            elif day in [0, 4]:
                price *= 1.10


        if 'car_class' in input_features:
            car_class = input_features['car_class']
            if car_class == 3:  #Премиум класс +40%
                price *= 1.4
            elif car_class == 2:  #Комфорт класс +20%
                price *= 1.2

        if 'driver_rating' in input_features:
            rating = input_features['driver_rating']
            if rating >= 4.9:  #Отличный рейтинг +15%
                price *= 1.15
            elif rating >= 4.7:  #Хороший рейтинг +8%
                price *= 1.08
            elif rating < 4.5:  #Низкий рейтинг -10%
                price *= 0.90

        if 'pickup_minutes' in input_features:
            pickup_time = input_features['pickup_minutes']
            if pickup_time < 5:  #Очень быстрая подача +10%
                price *= 1.10
            elif pickup_time > 15:  #Долгая подача -10%
                price *= 0.90

        if 'orders_last_hour' in input_features:
            orders = input_features['orders_last_hour']
            if orders > 20:  #Очень высокий спрос +25%
                price *= 1.25
            elif orders > 15:  #Высокий спрос +15%
                price *= 1.15
            elif orders < 5:   #Низкий спрос -10%
                price *= 0.90

        if 'avg_price_last_hour' in input_features:
            market_price = input_features['avg_price_last_hour']
            if market_price > price * 1.20:
                price = price * 1.15  #Поднимаем к рыночной
            elif market_price < price * 0.8:
                price = price * 0.95  #Опускаем к рыночной

        if 'platform_ios' in input_features and input_features['platform_ios'] == 1:
            price *= 1.03;

        price *= 1.01  # +1% для компании

        #Минимальная цена
        min_price = input_features.get('min_price', 200)
        price = max(price, min_price)
    
        #Максимальная цена
        max_price = input_features.get('max_price', 5000)
        price = min(price, max_price)

        #Округление до 10 руб
        price = round(price / 10) * 10

        return price

    def analyze_price_sensitivity(self, base_features, price_range=None):
        if price_range is None:
            price_range = list(range(200, 1001, 50))
        acceptance_rates = []
        if self.modeling_df is None or 'successful_price' not in self.modeling_df.columns:
            avg_price = self.modeling_df['successful_price'].mean()
        else:
            avg_price = 500

        for price in price_range:
            if price <= avg_price * 1.1:
                acceptance_rate = 0.8
            elif price <= avg_price * 1.3:
                acceptance_rate = 0.5
            else:
                acceptance_rate = 0.2
            acceptance_rates.append(acceptance_rate)

        return price_range, acceptance_rates


def main():
    st.set_page_config(page_title="Оптимизатор цен такси", page_icon="💰", layout="wide")

    st.markdown('<h1 style="text-align:center; color:#1f77b4">💰Оптимизатор цен такси</h1>', unsafe_allow_html=True)

    st.sidebar.header("📁 Загрузка исторических данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV с историей заказов", type=['csv'])

    if 'optimizer' not in st.session_state:
        st.session_state['optimizer'] = None
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'uploaded_name' not in st.session_state:
        st.session_state['uploaded_name'] = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Данные загружены: {df.shape}")

            if st.session_state['optimizer'] is None or st.session_state['uploaded_name'] != getattr(uploaded_file, 'name', None):
                st.session_state['optimizer'] = PriceOptimizer(df)
                st.session_state['model_trained'] = False
                st.session_state['uploaded_name'] = getattr(uploaded_file, 'name', None)

            optimizer = st.session_state['optimizer']

            #Автообучение
            auto_train = st.sidebar.checkbox("Автоматически обучить модель", value=True)
            if auto_train and not st.session_state['model_trained']:
                ok, info = optimizer.train_model()
                if ok:
                    mae, r2 = info
                    st.success(f"Модель автоматически обучена")
                    st.session_state['model_trained'] = True
                else:
                    st.info(info)

            #Ручного обучения
            if st.button("Обучить модель"):
                with st.spinner("Обучение модели на исторических данных..."):
                    ok, info = optimizer.train_model()
                    if ok:
                        mae, r2 = info
                        st.success(f"✅ Модель обучена! MAE={mae:.2f}, R²={r2:.3f}")
                        st.session_state['model_trained'] = True
                        if optimizer.feature_importance is not None and len(optimizer.feature_importance):
                            st.subheader("📊 Важность факторов в ценообразовании")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df = optimizer.feature_importance.head(10)
                            ax.barh(importance_df['feature'], importance_df['importance'])
                            ax.set_xlabel('Важность')
                            ax.set_title('Топ-10 факторов')
                            st.pyplot(fig)
                    else:
                        st.error(info)

            #Интерфейс нового заказа
            st.header("🎯 Расчет оптимальной цены для нового заказа")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Параметры заказа")
                distance = st.slider("Расстояние (км)", 1.0, 50.0, 5.0, 0.5)
                duration = st.slider("Примерное время поездки (мин)", 5, 120, 20, 1)
                pickup_time = st.slider("Время подачи (мин)", 1, 30, 5, 1)
                hour = st.slider("Час заказа", 0, 23, 12)
                day_of_week = st.selectbox("День недели", ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]) 

            with col2:
                st.subheader("Дополнительно")
                car_class = st.selectbox("Класс автомобиля", ["Эконом (LADA, ВАЗ)", "Комфорт (Kia, Hyundai)", "Премиум (Volkswagen, Skoda)"], index=0)
                driver_rating = st.slider("Рейтинг водителя", 4.0, 5.0, 4.8, 0.1)
                platform = st.selectbox("Платформа", ["android", "ios"], index=0)
                demand_level = st.select_slider("Уровень спроса", ["Очень низкий", "Низкий", "Средний", "Высокий", "Очень высокий"], value="Средний")

            day_map = {"Понедельник": 0, "Вторник": 1, "Среда": 2, "Четверг": 3, "Пятница": 4, "Суббота": 5, "Воскресенье": 6}
            car_class_map = {"Эконом (LADA, ВАЗ)": 1, "Комфорт (Kia, Hyundai)": 2, "Премиум (Volkswagen, Skoda)": 3}
            demand_map = {"Очень низкий": 0.5, "Низкий": 0.8, "Средний": 1.0, "Высокий": 1.3, "Очень высокий": 1.6}

            input_features = {
                'distance_km': distance,
                'duration_minutes': duration,
                'pickup_minutes': pickup_time,
                'hour': hour,
                'day_of_week': day_map[day_of_week],
                'is_weekend': 1 if day_map[day_of_week] in [5, 6] else 0,
                'is_rush_hour': 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
                'driver_rating': driver_rating,
                'car_class': car_class_map[car_class],
                'platform_android': 1 if platform == 'android' else 0,
                'platform_ios': 1 if platform == 'ios' else 0,
                'orders_last_hour': 10 * demand_map[demand_level],
                'avg_price_last_hour': 500 * demand_map[demand_level],
                'min_price': 150
            }

            if st.button("💰 Рассчитать оптимальную цену"):
                if st.session_state.get('model_trained', False) and st.session_state['optimizer'] is not None and st.session_state['optimizer'].model is not None:
                    try:
                        optimal_price = st.session_state['optimizer'].predict_optimal_price(input_features)

                        st.markdown(
                            f"<div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:20px; border-radius:12px; text-align:center;'>"
                            f"\n<h2>🎯 Оптимальная цена</h2>"
                            f"\n<h1>{int(optimal_price)} ₽</h1>"
                            f"\n<p>Расстояние: {distance} км | Время: {duration} мин</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        risk_colors = {
                            "Высокая вероятность согласия": "#16a34a",
                            "Почти нет риска отказа": "#65a30d",
                            "Оптимальная цена": "#2563eb",
                            "Умеренный риск отказа, но выше прибыль": "#f59e0b",
                            "Большой риск, клиент скорее всего откажется": "#dc2626"
                        }

                        # Блок с рисками отказа
                        risk_list = [
                            (optimal_price - 30, "Высокая вероятность согласия клиента"),
                            (optimal_price - 10, "Почти нет риска отказа"),
                            (optimal_price, "Оптимальная цена"),
                            (optimal_price + 20, "Умеренный риск отказа, но выше прибыль"),
                            (optimal_price + 30, "Большой риск, клиент скорее всего откажется"),
                        ]

                        risk_html = """
                        <div style='
                             background: linear-gradient(135deg, #f9fafb 0%, #eef2ff 100%);
                             border-radius: 16px;
                             padding: 18px;
                             margin-top: 15px;
                             box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                             text-align: center;
                             color: #333;
                        '>
                          <h3 style='margin-bottom:10px;'>💡 Оценка риска отказа</h3>
                          <p style='margin: 0 0 10px 0; font-size:15px; color:#555;'>
                             Чем выше цена, тем больше вероятность отказа пассажира от поездки.
                          </p>
                          <div style='display:flex; justify-content:center; flex-wrap:wrap; gap:12px; margin-top:10px;'>
                        """

                        # карта рисков с цветом и яркостью
                        risk_html = (
                            "<div style='background: linear-gradient(135deg, #f9fafb 0%, #eef2ff 100%);"
                            "border-radius: 16px; padding: 18px; margin-top: 15px;"
                            "box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center; color: #333;'>"
                            "<h3 style='margin-bottom:10px;'>💡 Оценка риска отказа</h3>"
                            "<p style='margin: 0 0 10px 0; font-size:15px; color:#555;'>"
                            "Чем выше цена, тем больше вероятность отказа пассажира от поездки.</p>"
                            "<div style='display:flex; justify-content:center; flex-wrap:wrap; gap:12px; margin-top:10px;'>"
                        )

                        for price, label in risk_list:
                            color = risk_colors.get(label, "#4b5563")
                            risk_html += (
                                f"<div style='background:{color}15; border: 1px solid {color};"
                                f"border-radius:10px; padding:10px 14px; min-width:180px;'>"
                                f"<span style='font-weight:600;color:{color};'>{int(price)} ₽</span>"
                                f"<p style='font-size:14px;color:{color};margin:4px 0 0 0;'>{label}</p>"
                                "</div>"
                            )
                        risk_html += "</div></div>"

                        st.markdown(risk_html, unsafe_allow_html=True)

                        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

                        # Анализ чувствительности
                        prices, acceptance = st.session_state['optimizer'].analyze_price_sensitivity(input_features)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))
                        ax1.plot(prices, acceptance, linewidth=2, marker='o')
                        ax1.axvline(optimal_price, color='red', linestyle='--', label=f'Оптимум: {int(optimal_price)}₽')
                        ax1.set_xlabel('Цена (₽)')
                        ax1.set_ylabel('Вероятность принятия')
                        ax1.set_title('Зависимость принятия от цены')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        if hasattr(st.session_state['optimizer'], 'modeling_df') and 'successful_price' in st.session_state['optimizer'].modeling_df.columns:
                            ax2.hist(st.session_state['optimizer'].modeling_df['successful_price'].dropna(), bins=30, edgecolor='black')
                            ax2.axvline(optimal_price, color='red', linestyle='--', linewidth=2)
                            ax2.set_xlabel('Цена (₽)')
                            ax2.set_ylabel('Количество заказов')
                            ax2.set_title('Сравнение с историческими ценами')
                            ax2.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        # Рекомендации и метрики
                        st.subheader("💡 Рекомендации")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Средняя историческая цена", f"{st.session_state['optimizer'].modeling_df['successful_price'].mean():.0f}₽")
                        with col2:
                            price_diff = optimal_price - st.session_state['optimizer'].modeling_df['successful_price'].mean()
                            st.metric("Отклонение от средней", f"{price_diff:+.0f}₽")
                        with col3:
                            if distance > 15:
                                st.metric("🏆 Добавка за дистанцию", "✅ Включена")
                            elif hour in [0, 1, 2, 3, 4, 5]:
                                st.metric("🌙 Ночной тариф", "✅ Активен")
                            else:
                                st.metric("⚡️ Условия", "📊 Стандартные")

                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
                else:
                    st.warning("Сначала обучите модель на исторических данных (или включите автообучение).")

            #Статистика
            if st.checkbox("Показать статистику данных"):
                st.subheader("📋 Статистика исторических данных")
                if hasattr(st.session_state['optimizer'], 'modeling_df'):
                    md = st.session_state['optimizer'].modeling_df
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Успешных заказов", len(st.session_state['optimizer'].success_df))
                        st.metric("Средняя цена", f"{st.session_state['optimizer'].success_df['successful_price'].mean():.0f}₽")
                    with col2:
                        st.metric("Мин. цена", f"{st.session_state['optimizer'].success_df['successful_price'].min():.0f}₽")
                        st.metric("Макс. цена", f"{st.session_state['optimizer'].success_df['successful_price'].max():.0f}₽")
                    with col3:
                        st.metric("Медианная цена", f"{st.session_state['optimizer'].success_df['successful_price'].median():.0f}₽")
                        st.metric("Стандартное отклонение", f"{st.session_state['optimizer'].success_df['successful_price'].std():.0f}₽")

        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
    else:
        st.info("👆 Загрузите CSV файл с историческими данными заказов такси")


if __name__ == "__main__":
    main()
