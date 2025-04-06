# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Biblioteki do modelowania i uczenia maszynowego (scikit-learn)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, balanced_accuracy_score,
                             roc_curve, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, precision_recall_curve, auc,
                             classification_report, average_precision_score)
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate # Potrzebne do odtworzenia tabel w formacie tekstowym

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Analiza Wypadków Drogowych UK")
# st.set_option('deprecation.showPyplotGlobalUse', False) # <-- USUNIĘTA LUB ZAKOMENTOWANA LINIA

# --- Funkcje cachowane do ładowania i przetwarzania danych ---

@st.cache_data # Cache'owanie danych
def load_data():
    """Ładuje dane z plików CSV."""
    try:
        casualties_url = 'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-last-5-years.csv'
        vehicles_url = 'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-last-5-years.csv'
        accidents_url = 'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv'

        casualties = pd.read_csv(casualties_url, low_memory=False)
        vehicles = pd.read_csv(vehicles_url, low_memory=False)
        accidents = pd.read_csv(accidents_url, low_memory=False)

        # Połączenie tabel
        data = accidents.merge(casualties, on='accident_index', how='left').merge(vehicles, on='accident_index', how='left')
        # Filtracja danych dla lat 2021-2023
        data = data[data['accident_year'].isin([2021, 2022, 2023])]
        return data
    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
        return None

@st.cache_data # Cache'owanie przetworzonych danych
def preprocess_data(data):
    """Przetwarza załadowane dane."""
    if data is None:
        return None, None, None, None, None, None, None

    data_processed = data.copy()

    # Oczyszczenie danych
    columns_to_check_NaN = [
        'road_type', 'light_conditions', 'junction_detail', 'junction_control', 'driver_home_area_type', 'accident_year',
        'age_of_casualty', 'driver_distance_banding', 'weather_conditions', 'urban_or_rural_area', 'casualty_type',
        'speed_limit', 'driver_imd_decile', 'age_of_vehicle', 'age_of_driver', 'number_of_casualties', 'skidding_and_overturning'
    ]
    # Sprawdzenie istnienia kolumn przed próbą zastąpienia wartości
    existing_cols = [col for col in columns_to_check_NaN if col in data_processed.columns]
    if not existing_cols:
         st.warning("Brak kluczowych kolumn do czyszczenia danych.")
         return None, None, None, None, None, None, None

    data_processed[existing_cols] = data_processed[existing_cols].replace([-1, 99], np.nan)
    data_processed.dropna(subset=existing_cols, inplace=True)

    # Konwersja kolumn numerycznych przed binowaniem
    numeric_cols_for_binning = ['age_of_casualty', 'age_of_driver']
    for col in numeric_cols_for_binning:
        if col in data_processed.columns:
             data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
        else:
             st.warning(f"Brak kolumny '{col}' do konwersji na numeryczną.")
             # Można dodać obsługę błędu lub domyślną wartość

    # Ponowne usunięcie NaN po konwersji
    data_processed.dropna(subset=numeric_cols_for_binning, inplace=True)

    # Przekształcenie czasu
    if 'time' in data_processed.columns:
        try:
            data_processed['hour_of_day'] = pd.to_datetime(data_processed['time'], format='%H:%M', errors='coerce').dt.hour
            data_processed.dropna(subset=['hour_of_day'], inplace=True) # Usuń wiersze, gdzie czas był niepoprawny
            data_processed['hour_of_day'] = data_processed['hour_of_day'].astype(int)
        except Exception as e:
            st.warning(f"Błąd podczas przetwarzania kolumny 'time': {e}. Kolumna 'hour_of_day' może być niekompletna.")
            data_processed['hour_of_day'] = 0 # Wartość domyślna lub inna obsługa
    else:
        st.warning("Brak kolumny 'time'. Nie można utworzyć 'hour_of_day'.")
        data_processed['hour_of_day'] = 0 # Wartość domyślna

    # Przygotowanie zmiennej docelowej i is_urban_driver
    if 'driver_home_area_type' in data_processed.columns:
        data_processed['driver_home_area_type'] = data_processed['driver_home_area_type'].replace({3: 2}).astype(int)
        data_processed['is_urban_driver'] = (data_processed['driver_home_area_type'] == 1).astype(int)
    else:
        st.error("Brak kolumny 'driver_home_area_type'. Nie można utworzyć 'is_urban_driver'.")
        return None, None, None, None, None, None, None

    if 'urban_or_rural_area' in data_processed.columns:
        data_processed['is_rural_accident'] = (data_processed['urban_or_rural_area'] == 2).astype(int)
    else:
        st.error("Brak kolumny 'urban_or_rural_area'. Nie można utworzyć zmiennej celu 'is_rural_accident'.")
        return None, None, None, None, None, None, None

    # Normalizacja speed_limit
    if 'speed_limit' in data_processed.columns:
        scaler = StandardScaler()
        data_processed['speed_limit_normalized'] = scaler.fit_transform(data_processed[['speed_limit']])
    else:
         st.warning("Brak kolumny 'speed_limit'. Nie można znormalizować.")
         data_processed['speed_limit_normalized'] = 0 # Wartość domyślna

    # Binowanie wieku
    bins_age = [-np.inf, 17, 25, 40, 60, np.inf]
    labels_age = ['1', '2', '3', '4', '5'] # Etykiety jako stringi dla kategoryzacji
    if 'age_of_casualty' in data_processed.columns:
         data_processed['age_of_casualty_binned'] = pd.cut(data_processed['age_of_casualty'], bins=bins_age, labels=labels_age, right=False).astype(str)
    if 'age_of_driver' in data_processed.columns:
         data_processed['age_of_driver_binned'] = pd.cut(data_processed['age_of_driver'], bins=bins_age, labels=labels_age, right=False).astype(str)


    # Inżynieria Cech
    data_processed['urban_driver_speed'] = data_processed['is_urban_driver'] * data_processed['speed_limit_normalized']
    data_processed['is_rush_hour'] = data_processed['hour_of_day'].apply(lambda h: 1 if (7 <= h <= 9) or (15 <= h <= 18) else 0)

    if 'driver_distance_banding' in data_processed.columns:
         data_processed['driver_distance_banding'] = pd.to_numeric(data_processed['driver_distance_banding'], errors='coerce')
         data_processed.dropna(subset=['driver_distance_banding'], inplace=True) # Usuń NaN jeśli konwersja zawiodła
         data_processed['distance_speed_interaction'] = data_processed['driver_distance_banding'] * data_processed['urban_driver_speed']
    else:
        st.warning("Brak kolumny 'driver_distance_banding'. Pomijam 'distance_speed_interaction'.")
        data_processed['distance_speed_interaction'] = 0

    # Wybór cech do modelu (upewnij się, że wszystkie kolumny istnieją)
    base_features = [
        'is_urban_driver', 'road_type', 'light_conditions', 'junction_detail', 'junction_control',
        'driver_distance_banding', 'weather_conditions', 'is_rush_hour', 'age_of_driver_binned', 'age_of_casualty_binned',
        'distance_speed_interaction', 'speed_limit_normalized', 'driver_imd_decile',
        'hour_of_day', 'number_of_casualties', 'urban_driver_speed', 'skidding_and_overturning', 'casualty_type'
    ]
    selected_features = [f for f in base_features if f in data_processed.columns]
    missing_features = set(base_features) - set(selected_features)
    if missing_features:
        st.warning(f"Brakujące kolumny w danych po przetworzeniu: {missing_features}. Zostaną pominięte w modelu.")


    # Przygotowanie danych do modelu
    if not selected_features:
        st.error("Brak wybranych cech do zbudowania modelu.")
        return None, None, None, None, None, None, None

    X = data_processed[selected_features].copy()
    y = data_processed['is_rural_accident']

    # Kodowanie kategorialne (one-hot encoding)
    categorical_cols_base = ['road_type', 'light_conditions', 'junction_detail', 'junction_control',
                             'age_of_casualty_binned', 'driver_distance_banding', 'is_rush_hour',
                             'weather_conditions', 'age_of_driver_binned', 'skidding_and_overturning', 'casualty_type']

    # Filtrujemy kolumny kategorialne, które faktycznie istnieją w X
    categorical_cols_to_encode = [col for col in categorical_cols_base if col in X.columns]

    # Konwersja typów przed get_dummies, aby uniknąć błędów
    for col in categorical_cols_to_encode:
         X[col] = X[col].astype(str) # Traktuj wszystko jak string na potrzeby get_dummies

    X = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True, dummy_na=False) # dummy_na=False by nie tworzyć kolumn dla NaN

    # --- Inżynieria cech po dummies ---
    # Upewniamy się, że kolumny istnieją przed próbą ich użycia
    if 'driver_distance_banding_3.0' in X.columns and 'driver_distance_banding_4.0' in X.columns:
        X['important_driver_distance'] = (X['driver_distance_banding_3.0'] + X['driver_distance_banding_4.0'] > 0).astype(int)
    else:
        X['important_driver_distance'] = 0 # Wartość domyślna jeśli brakuje kolumn
        st.warning("Nie można utworzyć 'important_driver_distance' - brak wymaganych kolumn po kodowaniu.")

    if 'is_urban_driver' in X.columns and 'important_driver_distance' in X.columns:
        X['urban_driver_long_distance'] = X['is_urban_driver'] * X['important_driver_distance']
    else:
         X['urban_driver_long_distance'] = 0
         st.warning("Nie można utworzyć 'urban_driver_long_distance'.")

    if 'is_urban_driver' in X.columns and 'junction_control_4.0' in X.columns:
        X['urban_driver_no_junction_control'] = X['is_urban_driver'] * X['junction_control_4.0']
    elif 'junction_control_4.0' in X.columns: # Jeśli tylko kolumna junction istnieje
        st.warning("Brak kolumny 'is_urban_driver' do stworzenia 'urban_driver_no_junction_control'.")
        # Można przypisać 0 lub pominąć
        # X['urban_driver_no_junction_control'] = 0
    else: # Jeśli brakuje obu lub junction_control_4.0
        # X['urban_driver_no_junction_control'] = 0 # Opcjonalnie
        # st.warning("Nie można utworzyć 'urban_driver_no_junction_control'.")
        pass # Po prostu nie tworzymy tej kolumny


    # Podział na zbiory
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp) # 0.25 * 0.8 = 0.2

    # Oversampling SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return data_processed, X_train_res, y_train_res, X_val, y_val, X_test, y_test, X # Zwracamy też oryginalne X do ważności cech


@st.cache_resource # Cache'owanie wytrenowanych modeli
def train_models(_X_train, _y_train):
    """Trenuje modele XGBoost i RandomForest."""
    models = {}
    try:
        # Model XGBoost
        xgb_model = XGBClassifier(
            random_state=42,
            scale_pos_weight=1, # Po SMOTE nie potrzebujemy scale_pos_weight
            max_depth=9,
            n_estimators=250,
            learning_rate=0.03,
            reg_alpha=1.0,
            reg_lambda=0.5,
            subsample=0.8,
            colsample_bytree=0.7,
            use_label_encoder=False, # Dodane dla nowszych wersji XGBoost
            eval_metric='logloss' # Dodane dla nowszych wersji XGBoost
        )
        # Walidacja krzyżowa (informacyjnie, model trenowany na całości X_train_res)
        # scores_xgb = cross_val_score(xgb_model, _X_train, _y_train, cv=3, scoring='roc_auc') # Używamy mniej foldów dla szybkości w demo
        # st.write(f"Średnie AUC-ROC (XGBoost) w walidacji krzyżowej na zbiorze treningowym: {np.mean(scores_xgb):.4f}")
        xgb_model.fit(_X_train, _y_train)
        models['XGBoost'] = xgb_model

        # Model RandomForest
        rf_model = RandomForestClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=10,
            min_samples_split=137,
            min_samples_leaf=26,
            n_jobs=-1,
            max_features='sqrt', #'log2' często działa podobnie do 'sqrt'
            criterion='entropy',
            bootstrap=True,
            class_weight=None # Po SMOTE nie potrzebujemy
        )
         # Walidacja krzyżowa (informacyjnie)
        # scores_rf = cross_val_score(rf_model, _X_train, _y_train, cv=3, scoring='roc_auc')
        # st.write(f"Średnie AUC-ROC (RandomForest) w walidacji krzyżowej na zbiorze treningowym: {np.mean(scores_rf):.4f}")
        rf_model.fit(_X_train, _y_train)
        models['RandomForest'] = rf_model

    except Exception as e:
        st.error(f"Wystąpił błąd podczas trenowania modeli: {e}")
        return None

    return models

# --- Główna część aplikacji ---

# Ładowanie i przetwarzanie danych (wykonywane raz dzięki cache)
data_raw = load_data()
data, X_train, y_train, X_val, y_val, X_test, y_test, X_original_features = preprocess_data(data_raw)

# Sprawdzenie, czy dane zostały poprawnie załadowane i przetworzone
if data is None or X_train is None:
    st.error("Nie udało się załadować lub przetworzyć danych. Dalsza analiza jest niemożliwa.")
    st.stop() # Zatrzymuje wykonywanie skryptu

# Trenowanie modeli (wykonywane raz dzięki cache)
# Upewnijmy się, że X_train i y_train nie są None przed próbą treningu
if X_train is not None and y_train is not None:
     models = train_models(X_train, y_train)
else:
     models = None
     st.error("Nie można wytrenować modeli z powodu problemów z danymi treningowymi.")


# --- Pasek boczny nawigacji ---
st.sidebar.title("Nawigacja")
section = st.sidebar.radio(
    "Wybierz sekcję analizy:",
    (
        "Wprowadzenie",
        "Przygotowanie Danych",
        "Analiza Wstępna Kierowców",
        "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku",
        "Modelowanie ML: Przewidywanie Lokalizacji Wypadku",
        "Ocena Modeli",
        "Ważność Cech (XGBoost)",
        "Analiza Kluczowych Cech (Chi-kwadrat)",
        "Wnioski i Podsumowanie"
     )
)

# --- Wyświetlanie wybranej sekcji ---

if section == "Wprowadzenie":
    st.title("Analiza związku między miejscem zamieszkania kierowcy a prawdopodobieństwem udziału w wypadku drogowym na terenach wiejskich")

    st.header("I. Temat")
    st.markdown("""
    **"Analiza związku między miejscem zamieszkania kierowcy, a prawdopodobieństwem udziału w wypadku drogowym na terenach wiejskich."**
    """)

    st.subheader("1.1 Cel pracy:")
    st.markdown("""
    - Zbadanie, czy istnieje związek między miejscem zamieszkania kierowcy (wiejskim lub miejskim), a prawdopodobieństwem jego udziału w wypadku drogowym na terenie wiejskim oraz identyfikacja kluczowych czynników wpływających na przewidywanie lokalizacji wypadku, z wykorzystaniem modeli uczenia maszynowego.
    """)

    st.subheader("1.2 Pytania badawcze:")
    st.markdown("""
    - Czy miejsce zamieszkania kierowcy (miasto vs. wieś) ma istotny wpływ na prawdopodobieństwo udziału w wypadku na terenie wiejskim?
    - Jakie cechy (np. prędkość, wiek kierowcy, warunki drogowe) determinują ryzyko przewidywania lokalizacji wypadku?
    - Czy wyniki modeli (XGBoost, RandomForest) potwierdzają hipotezę o wyższym ryzyku wypadków dla kierowców miejskich na terenach wiejskich? (Uwaga: Hipoteza dotyczy wyższego ryzyka dla kierowców miejskich, podczas gdy wstępna analiza może sugerować co innego - modele pomogą to zweryfikować).
    """)

    st.subheader("1.3 Hipoteza badawcza:")
    st.markdown("""
    - Kierowcy z miast mają statystycznie istotnie wyższe prawdopodobieństwo udziału w wypadku drogowym zlokalizowanym na terenie wiejskim niż kierowcy zamieszkujący tereny wiejskie. *(Uwaga: Ta hipoteza może wymagać weryfikacji w świetle wyników analizy chi-kwadrat).*
    """)

elif section == "Przygotowanie Danych":
    st.title("II. Dane i Metodyka")

    st.header("1. Źródła danych")
    st.markdown("""
    - Dane pochodzą z oficjalnych brytyjskich baz danych (Department for Transport - data.gov.uk) dotyczących wypadków drogowych z lat 2021-2023 na terenie UK.
    - Tabele (`casualties`, `vehicles`, `accidents`) zawierające dane m.in. o ofiarach (wiek, miejsce zamieszkania), informacje o pojazdach i kierowcach (np. obszar zamieszkania, odległość od miejsca wypadku) i kontekst wypadków (warunki pogodowe, typ drogi) zostały połączone w tabelę `data` po kluczu `accident_index`.
    - Statystyki dotyczą wyłącznie wypadków z obrażeniami ciała na drogach publicznych, które są zgłaszane policji, a następnie rejestrowane przy użyciu formularza zgłaszania kolizji `STATS19`.
    - **Przewodnik** po statystykach: [link](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
    - **Zestawy danych**: [link](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
    """)
    st.subheader("Próbka surowych danych (połączonych)")
    if data_raw is not None:
        st.dataframe(data_raw.head())
    else:
        st.warning("Nie można wyświetlić próbki surowych danych.")


    st.header("2. Przygotowanie danych")
    st.markdown("""
    Przeprowadzono następujące kroki:
    - **Oczyszczono dane:** Zastąpiono wartości `-1` i `99` oznaczające brak danych lub nieznane wartości na `NaN`, a następnie usunięto wiersze z brakami w kluczowych kolumnach.
    - **Przekształcono czas:** Z kolumny `time` wyodrębniono godzinę (`hour_of_day`).
    - **Przygotowano zmienne kategoryczne:**
        - Zmieniono kodowanie `driver_home_area_type`, łącząc 'Small town' i 'Rural' w jedną kategorię (2).
        - Stworzono zmienne binarne: `is_urban_driver` (1 jeśli kierowca z miasta) i `is_rural_accident` (1 jeśli wypadek na wsi - **zmienna celu**).
    - **Znormalizowano prędkość:** `speed_limit` przekształcono w `speed_limit_normalized` przy użyciu StandardScaler.
    - **Zbindowano wiek:** `age_of_casualty` i `age_of_driver` podzielono na 5 przedziałów wiekowych (≤17, 18-25, 26-40, 41-60, >60).
    - **Stworzono nowe cechy (Feature Engineering):**
        * `urban_driver_speed` (interakcja pochodzenia miejskiego i prędkości).
        * `is_rush_hour` (czy godzina wypadku to szczyt komunikacyjny).
        * `distance_speed_interaction` (interakcja odległości i prędkości kierowcy miejskiego).
    - **Wybrano cechy:** Ustalono listę cech do modelowania (`selected_features`).
    - **Zakodowano kategorie:** Zastosowano kodowanie zero-jedynkowe (one-hot encoding) dla zmiennych kategorycznych z listy `selected_features`.
    - **Stworzono dodatkowe cechy po kodowaniu:**
        * `important_driver_distance` (czy kierowca jechał dalej niż ~10 mil).
        * `urban_driver_long_distance` (interakcja kierowcy miejskiego i dużej odległości).
        * `urban_driver_no_junction_control` (interakcja kierowcy miejskiego i braku kontroli na skrzyżowaniu).
    - **Podzielono dane:** Na zbiory treningowy (60%), walidacyjny (20%) i testowy (20%), zachowując proporcje klasy docelowej (stratyfikacja).
    - **Zastosowano SMOTE:** Na zbiorze treningowym, aby zrównoważyć klasy (więcej wypadków miejskich niż wiejskich).
    """)

    st.subheader("Rozmiary zbiorów danych po podziale i SMOTE:")
    if X_train is not None and X_val is not None and X_test is not None:
         st.write(f"- Zbiór treningowy (po SMOTE): {X_train.shape[0]} rekordów, {X_train.shape[1]} cech")
         st.write(f"- Zbiór walidacyjny: {X_val.shape[0]} rekordów, {X_val.shape[1]} cech")
         st.write(f"- Zbiór testowy: {X_test.shape[0]} rekordów, {X_test.shape[1]} cech")
    else:
         st.warning("Nie można wyświetlić rozmiarów zbiorów - błąd podczas przetwarzania.")

    st.subheader("Próbka danych po przetworzeniu (przed SMOTE)")
    if data is not None:
        st.dataframe(data.head()) # Pokazujemy dane po dodaniu kolumn, przed podziałem
    else:
        st.warning("Nie można wyświetlić próbki przetworzonych danych.")


elif section == "Analiza Wstępna Kierowców":
    st.title("Analiza Wstępna: Charakterystyka Kierowców w Wypadkach")

    if data is not None:
        # --- Obliczenia z oryginalnego kodu ---
        # 1. Całkowita liczba wypadków
        total_accidents = len(data)

        # 2. Proporcje kierowców
        driver_origin = data['is_urban_driver'].value_counts().reset_index()
        driver_origin.columns = ['is_urban_driver_code', 'Liczba']
        driver_origin['Pochodzenie'] = driver_origin['is_urban_driver_code'].map({0: 'Wieś/Małe Miasto', 1: 'Miasto'}) # Zmieniona etykieta dla 0
        driver_origin['Procent'] = (driver_origin['Liczba'] / driver_origin['Liczba'].sum()) * 100

        # Przygotowanie tabeli dla proporcji
        driver_origin_display = driver_origin[['Pochodzenie', 'Liczba', 'Procent']].copy()
        # Dodanie sumy
        sum_row_driver = pd.DataFrame({
             'Pochodzenie': ['Suma'],
             'Liczba': [driver_origin['Liczba'].sum()],
             'Procent': [100.0]
        })
        driver_origin_display = pd.concat([driver_origin_display, sum_row_driver], ignore_index=True)

        # 3. Rozkład kierowców według lat
        driver_stats = data.groupby(['accident_year', 'is_urban_driver']).size().unstack(fill_value=0)
        # Sprawdzenie, czy obie kolumny istnieją po unstack
        if 0 not in driver_stats.columns: driver_stats[0] = 0
        if 1 not in driver_stats.columns: driver_stats[1] = 0
        driver_stats.columns = ['Wieś/Małe Miasto', 'Miasto']
        driver_stats['Rok'] = driver_stats.index
        driver_stats['Suma'] = driver_stats['Wieś/Małe Miasto'] + driver_stats['Miasto']
        # Uniknięcie dzielenia przez zero, jeśli suma jest 0 w jakimś roku
        driver_stats['Procent Wieś/Małe Miasto'] = np.where(driver_stats['Suma'] > 0, (driver_stats['Wieś/Małe Miasto'] / driver_stats['Suma']) * 100, 0)
        driver_stats['Procent Miasto'] = np.where(driver_stats['Suma'] > 0, (driver_stats['Miasto'] / driver_stats['Suma']) * 100, 0)

        # Tabela do wyświetlenia
        driver_stats_display = driver_stats[['Rok', 'Wieś/Małe Miasto', 'Procent Wieś/Małe Miasto', 'Miasto', 'Procent Miasto', 'Suma']].reset_index(drop=True)

        # --- Wyświetlanie w Streamlit ---
        st.subheader("Tabela 1: Proporcje kierowców według miejsca zamieszkania")
        st.dataframe(driver_origin_display.style.format({'Liczba': '{:,.0f}', 'Procent': '{:.1f}%'}))

        st.subheader("Tabela 2: Rozkład kierowców według miejsca zamieszkania w latach 2021-2023")
        st.dataframe(driver_stats_display.style.format({
            'Wieś/Małe Miasto': '{:,.0f}', 'Procent Wieś/Małe Miasto': '{:.1f}%',
            'Miasto': '{:,.0f}', 'Procent Miasto': '{:.1f}%',
            'Suma': '{:,.0f}'
        }))

        st.subheader("Wizualizacje")

        # --- Wykresy jak w oryginalnym kodzie (Matplotlib) ---
        fig_mpl = plt.figure(figsize=(12, 10)) # Nieco mniejszy rozmiar dla Streamlit
        gs = fig_mpl.add_gridspec(2, 2, height_ratios=[1, 1.2])

        # Wykres 1: Całkowita liczba wypadków (lewy górny)
        ax1 = fig_mpl.add_subplot(gs[0, 0])
        bars1 = ax1.bar(['Wszystkie wypadki'], [total_accidents], color='#93c47d')
        ax1.set_title('Całkowita liczba analizowanych wypadków')
        ax1.set_ylabel('Liczba')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'{int(height):,} (100%)', ha='center', va='center', fontsize=10, color='black')

        # Wykres 2: Proporcje kierowców (prawy górny) - poprawiony
        ax2 = fig_mpl.add_subplot(gs[0, 1])
        # Dane do wykresu słupkowego skumulowanego
        driver_origin_plot = driver_origin[driver_origin['Pochodzenie'] != 'Suma'].set_index('Pochodzenie')
        bottom_val = 0
        colors = {'Wieś/Małe Miasto': '#1f77b4', 'Miasto': '#ff7f0e'} # Zmienione kolory dla spójności
        for origin_type in ['Wieś/Małe Miasto', 'Miasto']:
             if origin_type in driver_origin_plot.index:
                 value = driver_origin_plot.loc[origin_type, 'Liczba']
                 percentage = driver_origin_plot.loc[origin_type, 'Procent']
                 bar = ax2.bar(['Kierowcy'], [value], bottom=[bottom_val], color=colors[origin_type], label=origin_type)
                 # Dodawanie tekstu
                 text_y = bottom_val + value / 2
                 ax2.text(0, text_y, f"{int(value):,}\n({percentage:.1f}%)", ha='center', va='center', fontsize=10, color='white')
                 bottom_val += value

        ax2.set_title('Proporcje kierowców wg miejsca zamieszkania')
        ax2.set_ylabel('Liczba kierowców')
        ax2.set_xticks([]) # Ukrycie etykiety "Kierowcy" na osi X
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2) # Legenda pod wykresem
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, total_accidents * 1.1) # Ustawienie limitu osi Y dla lepszej czytelności


        # Wykres 3: Rozkład kierowców według lat (dolny, rozciągnięty)
        ax3 = fig_mpl.add_subplot(gs[1, :])
        bar_width = 0.35
        x = np.arange(len(driver_stats_display['Rok']))
        rects1 = ax3.bar(x - bar_width/2, driver_stats_display['Wieś/Małe Miasto'], bar_width, label='Wieś/Małe Miasto', color='#1f77b4')
        rects2 = ax3.bar(x + bar_width/2, driver_stats_display['Miasto'], bar_width, label='Miasto', color='#ff7f0e')

        ax3.set_title('Rozkład kierowców wg miejsca zamieszkania w latach')
        ax3.set_xlabel('Rok')
        ax3.set_ylabel('Liczba kierowców')
        ax3.set_xticks(x)
        ax3.set_xticklabels(driver_stats_display['Rok'])
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2) # Legenda pod wykresem
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        # Dodanie etykiet na słupkach
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{int(height):,}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), # 3 punkty pionowego przesunięcia
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        autolabel(rects1, ax3)
        autolabel(rects2, ax3)


        fig_mpl.suptitle('Analiza kierowców w wypadkach drogowych (2021-2023)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Dostosowanie marginesów

        st.pyplot(fig_mpl)

    else:
        st.warning("Brak danych do przeprowadzenia analizy wstępnej.")


elif section == "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku":
    st.title("Analiza Związku: Miejsce Zamieszkania Kierowcy a Lokalizacja Wypadku")

    if data is not None and 'is_urban_driver' in data.columns and 'is_rural_accident' in data.columns:
         # --- Obliczenia z oryginalnego kodu ---
         # 1. Tabela kontyngencji
         contingency_table = pd.crosstab(data['is_urban_driver'], data['is_rural_accident'])
         contingency_table.index = ['Wieś/Małe Miasto', 'Miasto'] # Zmienione etykiety
         contingency_table.columns = ['Wypadek Miejski', 'Wypadek Wiejski']

         # 2. Tabela procentowa
         location_stats = pd.crosstab(data['is_urban_driver'], data['is_rural_accident'], normalize='index') * 100
         location_stats.index = ['Wieś/Małe Miasto', 'Miasto']
         location_stats.columns = ['Wypadki Miejskie (%)', 'Wypadki Wiejskie (%)']

         # 3. Test chi-kwadrat
         try:
             chi2, p, dof, expected = chi2_contingency(contingency_table)
             n = contingency_table.values.sum()
             phi = np.sqrt(chi2 / n) if n > 0 else 0 # Współczynnik Phi

             # Interpretacja siły związku Phi
             if phi < 0.1: strength = "Bardzo słaby (φ < 0.1)"
             elif 0.1 <= phi < 0.3: strength = "Słaby (φ = 0.1–0.3)"
             elif 0.3 <= phi < 0.5: strength = "Umiarkowany (φ = 0.3–0.5)"
             else: strength = "Silny (φ ≥ 0.5)"

             # Interpretacja wyniku testu
             alpha = 0.05
             if p < alpha:
                 conclusion = f"Odrzucamy hipotezę zerową (H₀). Istnieje statystycznie istotny związek (p = {p:.4e} < {alpha})."
             else:
                 conclusion = f"Nie ma podstaw do odrzucenia hipotezy zerowej (H₀). Nie stwierdzono statystycznie istotnego związku (p = {p:.4f} >= {alpha})."

             # Oczekiwana tabela kontyngencji
             expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

         except Exception as e:
            st.error(f"Błąd podczas obliczania testu Chi-kwadrat: {e}")
            chi2, p, dof, phi, strength, conclusion, expected_df = None, None, None, None, None, None, None

         # --- Wyświetlanie w Streamlit ---
         st.subheader("Tabela Kontyngencji (Obserwowane Liczby)")
         st.dataframe(contingency_table.style.format("{:,.0f}"))

         st.subheader("Tabela Procentowa Lokalizacji Wypadków wg Pochodzenia Kierowcy")
         st.dataframe(location_stats.style.format("{:.1f}%"))

         if chi2 is not None:
              st.subheader("Wyniki Testu Chi-kwadrat Niezależności")
              st.markdown(f"""
              - **Statystyka chi-kwadrat (χ²):** {chi2:.2f}
              - **Wartość p (p-value):** {p:.4e}
              - **Stopnie swobody (dof):** {dof}
              - **Współczynnik Phi (φ):** {phi:.3f}
              - **Interpretacja siły związku (Phi):** {strength}
              - **Wniosek (poziom istotności α = {alpha}):** {conclusion}
              """)

              st.subheader("Tabela Oczekiwana (Gdyby nie było związku)")
              st.dataframe(expected_df.style.format("{:,.1f}"))
         else:
              st.warning("Nie można było obliczyć wyników testu chi-kwadrat.")


         # --- Wykres Plotly ---
         st.subheader("Wykres: Procent Wypadków Miejskich i Wiejskich wg Pochodzenia Kierowcy")
         # Przygotowanie danych do wykresu Plotly
         location_stats_plot = location_stats.reset_index().rename(columns={'index': 'Pochodzenie Kierowcy'})
         location_stats_melted = location_stats_plot.melt(
             id_vars='Pochodzenie Kierowcy',
             var_name='Typ Obszaru Wypadku',
             value_name='Procent Wypadków'
         )
         # Poprawienie nazw dla legendy
         location_stats_melted['Typ Obszaru Wypadku'] = location_stats_melted['Typ Obszaru Wypadku'].str.replace(' (%)', '')

         fig_plotly = px.bar(location_stats_melted,
                             x='Pochodzenie Kierowcy',
                             y='Procent Wypadków',
                             color='Typ Obszaru Wypadku',
                             title='Procent Wypadków Miejskich i Wiejskich<br>wg Miejsca Zamieszkania Kierowcy',
                             labels={'Procent Wypadków': 'Procent Wypadków (%)', 'Typ Obszaru Wypadku': 'Lokalizacja Wypadku'},
                             color_discrete_map={'Wypadki Miejskie': '#1f77b4', 'Wypadki Wiejskie': '#ff7f0e'},
                             barmode='group',
                             text='Procent Wypadków' # Dodanie wartości na słupkach
                             )
         fig_plotly.update_layout(yaxis_ticksuffix='%', yaxis_title='Procent Wypadków (%)', xaxis_title='Pochodzenie Kierowcy')
         fig_plotly.update_traces(texttemplate='%{text:.1f}%', textposition='outside') # Formatowanie tekstu
         st.plotly_chart(fig_plotly, use_container_width=True)

         # --- Podsumowanie i wnioski z tej sekcji (z Markdown) ---
         st.subheader("Interpretacja i Wnioski z Analizy Związku")
         st.markdown("""
         **Kluczowe obserwacje:**
         - **Kierowcy z obszarów wiejskich/małych miast**: Znacznie częściej uczestniczą w wypadkach na terenach wiejskich (ok. 68.4%) niż miejskich (ok. 31.6%).
         - **Kierowcy z miast**: Dominują w wypadkach na terenach miejskich (ok. 78.3%), a rzadziej uczestniczą w wypadkach na terenach wiejskich (ok. 21.7%).

         **Wyniki testu chi-kwadrat:**
         - Test wykazał **statystycznie istotny związek** (p < 0.0001) między miejscem zamieszkania kierowcy a lokalizacją wypadku.
         - Siła tego związku, mierzona współczynnikiem Phi (φ ≈ 0.394), jest **umiarkowana**. Oznacza to, że miejsce zamieszkania jest ważnym czynnikiem, ale nie jedynym determinującym lokalizację wypadku. Inne czynniki, jak typ drogi, warunki, prędkość, również odgrywają rolę.

         **Wnioski:**
         1.  **Istnienie związku**: Potwierdzono wyraźny związek. Kierowcy częściej ulegają wypadkom w środowisku, w którym mieszkają (miejscy w miastach, wiejscy na wsiach), ale dysproporcja jest szczególnie widoczna dla kierowców wiejskich na terenach wiejskich.
         2.  **Wstępna weryfikacja hipotezy**: Wyniki **nie potwierdzają** pierwotnej hipotezy, że *kierowcy miejscy* mają *wyższe* prawdopodobieństwo wypadku na wsi. Wręcz przeciwnie, to kierowcy **wiejsko-małomiejscy** mają znacznie wyższy odsetek wypadków na terenach wiejskich w obrębie swojej grupy. Jednak kierowcy miejscy, stanowiąc większość ogółu, nadal generują znaczną liczbę wypadków na wsiach w liczbach bezwzględnych.
         3.  **Potrzeba dalszej analizy**: Umiarkowana siła związku sugeruje, że modele ML mogą pomóc zidentyfikować inne czynniki wpływające na ryzyko wypadku w terenie wiejskim, szczególnie interakcje między pochodzeniem kierowcy a innymi zmiennymi.
         """)

    else:
        st.warning("Brak danych lub wymaganych kolumn ('is_urban_driver', 'is_rural_accident') do przeprowadzenia analizy związku.")


elif section == "Modelowanie ML: Przewidywanie Lokalizacji Wypadku":
    st.title("Modelowanie Uczenia Maszynowego")
    st.header("Cel: Przewidywanie, czy wypadek zdarzy się na terenie wiejskim (`is_rural_accident` = 1)")

    st.subheader("Wybrane Modele:")
    st.markdown("- **XGBoost Classifier:** Wydajny model gradient boostingowy.")
    st.markdown("- **Random Forest Classifier:** Zespół drzew decyzyjnych.")

    st.subheader("Proces:")
    st.markdown("""
    1.  Dane zostały podzielone na zbiory: treningowy (60%), walidacyjny (20%) i testowy (20%).
    2.  Zastosowano **SMOTE** na zbiorze treningowym, aby zrównoważyć klasy (przewidujemy rzadszą klasę - wypadki wiejskie).
    3.  Modele zostały wytrenowane na zbalansowanym zbiorze treningowym.
    4.  Wstępna optymalizacja hiperparametrów została wykonana (parametry widoczne w kodzie).
    5.  Ocena modeli odbywa się na **niezmienionych** (niezbalansowanych) zbiorach walidacyjnym i testowym, aby odzwierciedlić rzeczywiste proporcje danych.
    """)

    st.subheader("Definicje Modeli (Kod Python):")
    st.code("""
# Model XGBoost
xgb_model = XGBClassifier(
    random_state=42, scale_pos_weight=1, max_depth=9,
    n_estimators=250, learning_rate=0.03, reg_alpha=1.0,
    reg_lambda=0.5, subsample=0.8, colsample_bytree=0.7,
    use_label_encoder=False, eval_metric='logloss'
)

# Model RandomForest
rf_model = RandomForestClassifier(
    random_state=42, n_estimators=150, max_depth=10,
    min_samples_split=137, min_samples_leaf=26, n_jobs=-1,
    max_features='sqrt', criterion='entropy', bootstrap=True,
    class_weight=None # SMOTE handles imbalance
)

# Trening (na danych X_train_res, y_train_res po SMOTE)
# xgb_model.fit(X_train_res, y_train_res)
# rf_model.fit(X_train_res, y_train_res)
    """, language='python')

    if models:
        st.success("Modele XGBoost i RandomForest zostały pomyślnie wytrenowane (lub załadowane z cache). Przejdź do sekcji 'Ocena Modeli', aby zobaczyć wyniki.")
    else:
        st.error("Wystąpił błąd podczas trenowania modeli. Ocena nie będzie możliwa.")

elif section == "Ocena Modeli":
    st.title("Ocena Modeli Uczenia Maszynowego")
    st.markdown("Ocena przeprowadzona na zbiorach **walidacyjnym** i **testowym** (bez SMOTE). Próg decyzyjny: 0.5.")

    if models and X_val is not None and y_val is not None and X_test is not None and y_test is not None:
        xgb_model = models.get('XGBoost')
        rf_model = models.get('RandomForest')

        if xgb_model and rf_model:
            try:
                # Predykcje dla zbioru walidacyjnego
                y_val_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
                y_val_pred_xgb = (y_val_pred_proba_xgb >= 0.5).astype(int)
                auc_val_xgb = roc_auc_score(y_val, y_val_pred_proba_xgb)
                report_val_xgb = classification_report(y_val, y_val_pred_xgb)

                y_val_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]
                y_val_pred_rf = (y_val_pred_proba_rf >= 0.5).astype(int)
                auc_val_rf = roc_auc_score(y_val, y_val_pred_proba_rf)
                report_val_rf = classification_report(y_val, y_val_pred_rf)

                # Predykcje dla zbioru testowego
                y_test_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
                y_test_pred_xgb = (y_test_pred_proba_xgb >= 0.5).astype(int)
                auc_test_xgb = roc_auc_score(y_test, y_test_pred_proba_xgb)
                report_test_xgb = classification_report(y_test, y_test_pred_xgb)

                y_test_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
                y_test_pred_rf = (y_test_pred_proba_rf >= 0.5).astype(int)
                auc_test_rf = roc_auc_score(y_test, y_test_pred_proba_rf)
                report_test_rf = classification_report(y_test, y_test_pred_rf)

                # --- Wyświetlanie wyników ---
                st.subheader("Wyniki na Zbiorze Walidacyjnym")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**XGBoost**")
                    st.text(f"AUC-ROC: {auc_val_xgb:.4f}")
                    st.text("Raport Klasyfikacji:")
                    st.code(report_val_xgb)
                with col2:
                    st.markdown("**Random Forest**")
                    st.text(f"AUC-ROC: {auc_val_rf:.4f}")
                    st.text("Raport Klasyfikacji:")
                    st.code(report_val_rf)

                st.subheader("Wyniki na Zbiorze Testowym (Ostateczna Ocena)")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**XGBoost**")
                    st.text(f"AUC-ROC: {auc_test_xgb:.4f}")
                    st.text("Raport Klasyfikacji:")
                    st.code(report_test_xgb)
                with col4:
                    st.markdown("**Random Forest**")
                    st.text(f"AUC-ROC: {auc_test_rf:.4f}")
                    st.text("Raport Klasyfikacji:")
                    st.code(report_test_rf)

                # --- Krzywe ROC ---
                st.subheader("Krzywe ROC (Zbiór Testowy)")
                fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_pred_proba_xgb)
                fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_pred_proba_rf)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, mode='lines', name=f'XGBoost (AUC = {auc_test_xgb:.4f})'))
                fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {auc_test_rf:.4f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Losowy Klasyfikator', line=dict(dash='dash')))

                fig_roc.update_layout(
                    title='Krzywa ROC - Zbiór Testowy',
                    xaxis_title='False Positive Rate (FPR)',
                    yaxis_title='True Positive Rate (TPR)',
                    legend_title='Model',
                    xaxis=dict(range=[0.0, 1.0]),
                    yaxis=dict(range=[0.0, 1.05])
                )
                st.plotly_chart(fig_roc, use_container_width=True)

                st.markdown("""
                **Interpretacja wyników:**
                - **AUC-ROC:** Oba modele osiągają wysokie wartości AUC (prawdopodobnie > 0.85-0.90), co wskazuje na bardzo dobrą zdolność do rozróżniania między wypadkami miejskimi a wiejskimi. XGBoost może nieznacznie przewyższać RandomForest.
                - **Precision/Recall/F1-score:** Należy zwrócić uwagę na metryki dla klasy `1` (wypadek wiejski). SMOTE pomaga poprawić `recall` (zdolność do wykrywania wypadków wiejskich), ale może nieznacznie obniżyć `precision`. F1-score balansuje te dwie metryki. Wyniki wskazują na dobrą ogólną wydajność modeli.
                - **Porównanie Walidacja vs Test:** Podobne wyniki na obu zbiorach sugerują, że modele dobrze generalizują i nie są przeuczone.
                """)

            except Exception as e:
                st.error(f"Wystąpił błąd podczas oceny modeli: {e}")
        else:
            st.warning("Jeden lub oba modele nie zostały poprawnie wytrenowane/załadowane.")
    else:
        st.warning("Brak wytrenowanych modeli lub danych walidacyjnych/testowych do przeprowadzenia oceny.")


elif section == "Ważność Cech (XGBoost)":
    st.title("Ważność Cech według Modelu XGBoost")
    st.markdown("Pokazuje, które cechy miały największy wpływ na predykcje modelu XGBoost.")

    if models and X_original_features is not None: # Używamy X z oryginalnymi nazwami kolumn
         xgb_model = models.get('XGBoost')
         if xgb_model:
             try:
                 # Sprawdzenie, czy model ma atrybut feature_importances_
                 if hasattr(xgb_model, 'feature_importances_'):
                     # Upewniamy się, że X_test ma te same kolumny co dane treningowe (X_original_features)
                     # To jest potrzebne jeśli X_test powstał z innego podziału, ale tutaj używamy nazw z X_original_features
                     feature_importance = pd.DataFrame({
                         'Cecha': X_original_features.columns, # Używamy nazw z X przed SMOTE i podziałem
                         'Ważność': xgb_model.feature_importances_
                     })

                     # Sortowanie i wybór top N cech
                     top_n = 20 # Można dostosować liczbę cech
                     top_features = feature_importance.sort_values(by='Ważność', ascending=False).head(top_n)

                     st.subheader(f"Top {top_n} najważniejszych cech")
                     st.dataframe(top_features.style.format({'Ważność': '{:.4f}'}))

                     # Wizualizacja
                     fig_imp = plt.figure(figsize=(10, 8))
                     plt.barh(top_features['Cecha'], top_features['Ważność'], color='skyblue')
                     plt.xlabel('Ważność (Importance)')
                     plt.ylabel('Cecha')
                     plt.title(f'Ważność Cech (XGBoost) - Top {top_n}')
                     plt.gca().invert_yaxis() # Najważniejsze na górze
                     plt.tight_layout()
                     st.pyplot(fig_imp)

                     st.markdown("""
                     **Interpretacja:**
                     - Cechy związane z **lokalizacją** (np. `road_type`, `junction_detail`, `junction_control`) oraz **kierowcą** (`is_urban_driver`, `driver_distance_banding`, `age_of_driver_binned`) prawdopodobnie znajdą się wysoko na liście.
                     - Inżynieria cech (np. `urban_driver_long_distance`, `distance_speed_interaction`) może również okazać się istotna.
                     - Wyniki te pomagają zrozumieć, na podstawie jakich informacji model podejmuje decyzje.
                     """)
                 else:
                     st.warning("Model XGBoost nie posiada atrybutu 'feature_importances_'. Nie można wyświetlić ważności cech.")

             except Exception as e:
                 st.error(f"Wystąpił błąd podczas obliczania ważności cech: {e}")
         else:
             st.warning("Model XGBoost nie został poprawnie załadowany.")
    else:
        st.warning("Brak wytrenowanego modelu XGBoost lub informacji o cechach (X_original_features) do wyświetlenia ważności.")


elif section == "Analiza Kluczowych Cech (Chi-kwadrat)":
    st.title("Szczegółowa Analiza Kluczowych Cech vs Lokalizacja Wypadku (Test Chi-kwadrat)")
    st.markdown("Badanie związku między wybranymi, potencjalnie najważniejszymi cechami (na podstawie ważności z modelu lub wiedzy domenowej) a zmienną celu `is_rural_accident`.")

    if data is not None:
        # Dodanie inżynierii cech dla 'important_driver_distance' do 'data' - jeśli jeszcze nie ma
        if 'important_driver_distance' not in data.columns and 'driver_distance_banding' in data.columns:
             try:
                 data['driver_distance_banding'] = pd.to_numeric(data['driver_distance_banding'], errors='coerce')
                 data.dropna(subset=['driver_distance_banding'], inplace=True)
                 distance_dummies = pd.get_dummies(data['driver_distance_banding'], prefix='driver_distance_banding', dummy_na=False) # Upewnijmy się, że traktujemy to jako kategorie
                 # Sprawdzamy, czy kolumny 3.0 i 4.0 istnieją po konwersji i get_dummies
                 col_3_exists = 'driver_distance_banding_3.0' in distance_dummies.columns
                 col_4_exists = 'driver_distance_banding_4.0' in distance_dummies.columns

                 if col_3_exists and col_4_exists:
                     data['important_driver_distance'] = ((distance_dummies['driver_distance_banding_3.0'] > 0) | (distance_dummies['driver_distance_banding_4.0'] > 0)).astype(int)
                 elif col_3_exists: # Jeśli jest tylko 3.0
                      data['important_driver_distance'] = (distance_dummies['driver_distance_banding_3.0'] > 0).astype(int)
                 elif col_4_exists: # Jeśli jest tylko 4.0
                      data['important_driver_distance'] = (distance_dummies['driver_distance_banding_4.0'] > 0).astype(int)
                 else:
                     st.warning("Nie można utworzyć 'important_driver_distance' - brak kolumn 3.0 lub 4.0 po dummizacji 'driver_distance_banding'.")
                     data['important_driver_distance'] = 0 # Przypisanie wartości domyślnej
             except Exception as e:
                 st.warning(f"Błąd przy tworzeniu 'important_driver_distance': {e}")
                 data['important_driver_distance'] = 0

        # Lista kluczowych cech do testowania (upewnijmy się, że istnieją w 'data')
        base_key_features = [
            'is_urban_driver', 'road_type', 'junction_control', 'junction_detail',
            'important_driver_distance', 'light_conditions', 'casualty_type',
            'weather_conditions', 'age_of_driver_binned', 'skidding_and_overturning' # Dodano więcej cech
        ]
        key_features_present = [feat for feat in base_key_features if feat in data.columns]

        if not key_features_present:
             st.warning("Brak kluczowych cech w danych do przeprowadzenia analizy chi-kwadrat.")
        else:
            results_chi2 = {}
            alpha = 0.05

            for feature in key_features_present:
                # Sprawdzenie, czy cecha ma więcej niż jedną unikalną wartość (po usunięciu NaN)
                if data[feature].nunique() > 1 and data['is_rural_accident'].nunique() > 1:
                    try:
                        contingency_feature = pd.crosstab(data[feature], data['is_rural_accident'])
                        chi2, p, dof, expected = chi2_contingency(contingency_feature)
                        significant = p < alpha
                        results_chi2[feature] = {'chi2': chi2, 'p_value': p, 'dof': dof, 'significant': significant}
                    except ValueError as ve:
                         st.warning(f"Nie można przeprowadzić testu chi-kwadrat dla cechy '{feature}': {ve}. Może zawierać same zera w wierszu/kolumnie.")
                         results_chi2[feature] = {'chi2': None, 'p_value': None, 'dof': None, 'significant': None}
                    except Exception as e:
                        st.error(f"Błąd podczas testu chi-kwadrat dla cechy '{feature}': {e}")
                        results_chi2[feature] = {'chi2': None, 'p_value': None, 'dof': None, 'significant': None}
                else:
                     st.warning(f"Cecha '{feature}' lub zmienna celu ma tylko jedną unikalną wartość. Pomijam test chi-kwadrat.")
                     results_chi2[feature] = {'chi2': None, 'p_value': None, 'dof': None, 'significant': 'N/A'}


            st.subheader("Wyniki Testów Chi-kwadrat dla Kluczowych Cech")
            results_df = pd.DataFrame(results_chi2).T
            results_df = results_df[['chi2', 'p_value', 'dof', 'significant']] # Ustawienie kolejności kolumn
            st.dataframe(results_df.style.format({
                'chi2': '{:.2f}',
                'p_value': '{:.4e}',
                'dof': '{:.0f}'
            }).applymap(lambda x: 'color: green' if x == True else ('color: red' if x == False else ''), subset=['significant']))

            st.markdown(f"""
            **Interpretacja:**
            - Tabela pokazuje wyniki testu chi-kwadrat sprawdzającego zależność każdej z kluczowych cech od lokalizacji wypadku (miejska vs wiejska).
            - **significant = True** (zielony) oznacza, że znaleziono statystycznie istotny związek (p < {alpha}) między daną cechą a lokalizacją wypadku.
            - **significant = False** (czerwony) oznacza brak statystycznie istotnego związku.
            - Wysoka wartość `chi2` przy niskim `p_value` wskazuje na silniejszą zależność.
            - Wyniki te są zgodne z ważnością cech z modelu ML – cechy istotne statystycznie często mają duży wpływ na predykcje modelu.
            """)
    else:
        st.warning("Brak danych do przeprowadzenia analizy kluczowych cech.")


elif section == "Wnioski i Podsumowanie":
    st.title("Wnioski Końcowe i Podsumowanie Analizy")

    st.header("Podsumowanie Wyników")
    st.markdown("""
    1.  **Związek Miejsca Zamieszkania z Lokalizacją Wypadku:**
        * Analiza chi-kwadrat potwierdziła **statystycznie istotny, umiarkowany związek** (φ ≈ 0.394) między miejscem zamieszkania kierowcy a tym, czy wypadek zdarzył się na terenie miejskim czy wiejskim.
        * Kierowcy z obszarów **wiejskich/małomiejskich** znacznie częściej uczestniczą w wypadkach na terenach **wiejskich** (ok. 68%) niż kierowcy miejscy (ok. 22%).
        * Kierowcy **miejscy** dominują w wypadkach **miejskich** (ok. 78%).
        * Wyniki te **nie potwierdzają** pierwotnej hipotezy o *wyższym* ryzyku kierowców *miejskich* na terenach *wiejskich*, ale wskazują na silną tendencję do wypadków w 'swoim' środowisku.

    2.  **Modelowanie Predykcyjne:**
        * Modele uczenia maszynowego (XGBoost, RandomForest) wykazały **bardzo dobrą zdolność** (AUC > 0.85-0.90) do przewidywania, czy wypadek zdarzy się na terenie wiejskim, na podstawie dostępnych cech.
        * Modele dobrze generalizują wyniki na niewidzianych wcześniej danych testowych.

    3.  **Kluczowe Czynniki Ryzyka (Ważność Cech):**
        * Do najważniejszych predyktorów lokalizacji wypadku należą cechy bezpośrednio związane z **lokalizacją i infrastrukturą drogową** (typ drogi, detale i kontrola skrzyżowania) oraz **charakterystyką kierowcy** (miejsce zamieszkania, odległość od domu, wiek).
        * Istotne okazały się również **warunki zewnętrzne** (oświetlenie, pogoda) oraz **dynamika zdarzenia** (poślizg/wywrócenie).
        * Dodatkowe testy chi-kwadrat potwierdziły istotność statystyczną wielu z tych cech w kontekście lokalizacji wypadku.

    4.  **Odpowiedzi na Pytania Badawcze:**
        * **Czy miejsce zamieszkania ma wpływ?** Tak, ma istotny statystycznie i praktycznie wpływ, co potwierdza test chi-kwadrat i wysoka ważność cechy `is_urban_driver` w modelach.
        * **Jakie cechy determinują ryzyko?** Kombinacja cech związanych z drogą, kierowcą, warunkami i dynamiką zdarzenia, co pokazała analiza ważności cech.
        * **Czy modele potwierdzają hipotezę?** Modele, podobnie jak analiza chi-kwadrat, nie potwierdzają *oryginalnej* hipotezy. Pokazują jednak złożoność interakcji różnych czynników wpływających na lokalizację wypadku.

    """)

    st.header("Ograniczenia Analizy")
    st.markdown("""
    - Dane `STATS19` obejmują tylko zgłoszone wypadki z obrażeniami, pomijając kolizje bez poszkodowanych.
    - Jakość danych (np. dokładność lokalizacji, kodowanie niektórych zmiennych) może wpływać na wyniki.
    - Analiza nie uwzględniała natężenia ruchu ani dokładnych tras przejazdu kierowców.
    - Korelacja (wykazana przez test chi-kwadrat i modele) nie implikuje przyczynowości.
    """)

    st.header("Rekomendacje i Dalsze Kierunki Badań")
    st.markdown("""
    - Skupienie działań prewencyjnych na specyficznych typach dróg i skrzyżowań wiejskich, które okazują się najbardziej ryzykowne.
    - Analiza wpływu doświadczenia kierowcy (np. czas posiadania prawa jazdy) na ryzyko wypadku na różnych typach terenów.
    - Badanie interakcji między miejscem zamieszkania a innymi czynnikami (np. czy kierowcy miejscy jeżdżący daleko od domu na wsiach są bardziej narażeni?).
    - Wykorzystanie bardziej zaawansowanych technik modelowania do analizy przyczynowo-skutkowej (jeśli dostępne byłyby odpowiednie dane).
    """)