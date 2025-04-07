# app_static.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Analiza Wypadków Drogowych UK (Statyczna)")

# --- Pasek boczny nawigacji ---
st.sidebar.title("Nawigacja")
section = st.sidebar.radio(
    "Wybierz sekcję analizy:",
    (
        "Wprowadzenie",
        "Opis Przygotowania Danych",
        "Analiza Wstępna Kierowców",
        "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku",
        "Opis Modelowania ML",
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
    - Czy miejsce zamieszkania kierowcy (miejskie vs. wiejskie) wpływa na prawdopodobieństwo udziału w wypadku drogowym na terenie wiejskim?
    - Jakie z wybranych cech kontekstowych (np. typ drogi, warunki oświetleniowe, kontrola skrzyżowań) mają największy wpływ na prawdopodobieństwo wystąpienia wypadku na terenie wiejskim?
    - Czy modele uczenia maszynowego (XGBoost, RandomForest) mogą skutecznie przewidzieć lokalizację wypadku na podstawie miejsca zamieszkania kierowcy i cech kontekstowych?
    """)

    st.subheader("1.3 Hipoteza badawcza:")
    st.markdown("""
    - Kierowcy z obszarów miejskich są bardziej narażeni na udział w wypadkach drogowych na terenach wiejskich niż kierowcy z obszarów wiejskich.
    - Specyficzne cechy, takie jak drogi jednopasmowe, brak oświetlenia ulicznego oraz niekontrolowane skrzyżowania, znacząco zwiększają ryzyko wypadku na terenie wiejskim.
    - Modele uczenia maszynowego (XGBoost, RandomForest) nie osiągają wysokiej skuteczności w przewidywaniu lokalizacji wypadku (wiejskiej vs. miejskiej) na podstawie miejsca zamieszkania kierowcy i cech kontekstowych.
    """)

elif section == "Opis Przygotowania Danych":
    st.title("II. Dane i Metodyka - Opis Przygotowania Danych")

    st.header("1. Źródła danych")
    st.markdown("""
    - Dane pochodzą z oficjalnych brytyjskich baz danych (Department for Transport - data.gov.uk) dotyczących wypadków drogowych z lat 2021-2023 na terenie UK.
    - Tabele (`casualties`, `vehicles`, `accidents`) zawierające dane m.in. o ofiarach (wiek, miejsce zamieszkania), informacje o pojazdach i kierowcach (np. obszar zamieszkania, odległość od miejsca wypadku) oraz kontekst wypadków (warunki pogodowe, typ drogi) zostały połączone w tabelę `data` po kluczu `accident_index`.
    - Statystyki dotyczą wyłącznie wypadków z obrażeniami ciała na drogach publicznych, które są zgłaszane policji, a następnie rejestrowane przy użyciu formularza zgłaszania kolizji `STATS19`.
    - **Przewodnik** po statystykach dotyczących wypadków drogowych: [link](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
    - **Zestawy danych** do pobrania: [link](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
    """)

    st.header("2. Opis kroków przygotowania danych")
    st.markdown("""
    W oryginalnej analizie przeprowadzono następujące kroki przygotowania danych (nie są one wykonywane w tej statycznej wersji):

    - **Filtrowanie danych:** Wybrano dane z lat 2021-2023.
    - **Oczyszczenie danych:** Zastąpiono wartości `-1` i `99` na `NaN` w kluczowych kolumnach, a następnie usunięto wiersze z brakami w tych kolumnach. Oczyszczone kolumny obejmują: `road_type`, `light_conditions`, `junction_detail`, `junction_control`, `driver_home_area_type`, `accident_year`, `age_of_casualty`, `driver_distance_banding`, `weather_conditions`, `urban_or_rural_area`, `casualty_type`, `speed_limit`, `driver_imd_decile`, `age_of_vehicle`, `age_of_driver`, `number_of_casualties`, `skidding_and_overturning`.
    - **Przekształcenie czasu:** Z kolumny `time` wyodrębniono godzinę i utworzono nową kolumnę `hour_of_day`.
    - **Przygotowanie zmiennych kategorycznych:**
      - Dla `driver_home_area_type` zsumowano wartości 2 (small town) i 3 (rural) w jedną etykietę o wartości 2, aby uprościć dane.
      - Stworzono zmienną binarną `is_urban_driver` (1 = kierowca z obszaru miejskiego, gdy `driver_home_area_type` = 1; 0 w przeciwnym razie).
      - Stworzono zmienną docelową `is_rural_accident` (1 = wypadek na terenie wiejskim, gdy `urban_or_rural_area` = 2; 0 w przeciwnym razie).
    - **Normalizacja:** Kolumnę `speed_limit` znormalizowano za pomocą `StandardScaler`, tworząc `speed_limit_normalized`.
    - **Binowanie wieku:** Kolumny `age_of_casualty` i `age_of_driver` podzielono na 5 przedziałów wiekowych (≤17 lat, 18-25 lat, 26-40 lat, 41-60 lat, >60 lat), tworząc odpowiednio `age_of_casualty_binned` i `age_of_driver_binned`.
    - **Inżynieria cech:**
      - `urban_driver_speed` jako iloczyn `is_urban_driver` i `speed_limit_normalized`.
      - `is_rush_hour` na podstawie `hour_of_day` (1, jeśli godzina należy do godzin szczytu: 7:00-9:00 lub 15:00-18:00; 0 w przeciwnym razie).
      - `distance_speed_interaction` jako iloczyn `driver_distance_banding` i `urban_driver_speed`.
    - **Wybór cech:** Ustalono listę `selected_features`, obejmującą: `is_urban_driver`, `road_type`, `light_conditions`, `junction_detail`, `junction_control`, `driver_distance_banding`, `weather_conditions`, `is_rush_hour`, `age_of_driver_binned`, `age_of_casualty_binned`, `distance_speed_interaction`, `speed_limit_normalized`, `driver_imd_decile`, `hour_of_day`, `number_of_casualties`, `urban_driver_speed`, `skidding_and_overturning`, `casualty_type`.
    - **Kodowanie kategoryczne:** Zmienne kategoryczne z `selected_features` (`road_type`, `light_conditions`, `junction_detail`, `junction_control`, `age_of_casualty_binned`, `driver_distance_banding`, `is_rush_hour`, `weather_conditions`, `age_of_driver_binned`, `skidding_and_overturning`, `casualty_type`) zakodowano metodą zero-jedynkową (one-hot encoding) z użyciem `pd.get_dummies`.
    - **Dodatkowe cechy po kodowaniu:**
      - `important_driver_distance` jako zmienna binarna (1, jeśli `driver_distance_banding_4.0` lub `driver_distance_banding_3.0` > 0, czyli dystans > 20 km; 0 w przeciwnym razie).
      - `urban_driver_long_distance` jako iloczyn `is_urban_driver` i `important_driver_distance`.
      - `urban_driver_no_junction_control` jako iloczyn `is_urban_driver` i `junction_control_4.0` (jeśli taka kolumna istnieje po kodowaniu, co oznacza brak kontroli ruchu na skrzyżowaniu).
    - **Podział danych:** Dane podzielono na zbiory:
      - Treningowy + walidacyjny (80%) i testowy (20%) z zachowaniem stratyfikacji.
      - Następnie zbiór treningowy + walidacyjny podzielono na treningowy (60% całości) i walidacyjny (20% całości), również ze stratyfikacją.
    - **Balansowanie danych:** Zastosowano SMOTE na zbiorze treningowym, aby zrównoważyć klasy zmiennej docelowej `is_rural_accident`.

    Celem było przygotowanie danych (X) i zmiennej docelowej (y, czyli `is_rural_accident`) do modelowania poprzez oczyszczenie, transformację i stworzenie nowych cech, uwzględniając również typ uczestnika wypadku (`casualty_type`).

    *Ta wersja aplikacji jedynie **prezentuje** wyniki uzyskane po tych krokach.*
    """)

    st.subheader("Rozmiary zbiorów danych po przetworzeniu:")
    st.write(f"- Zbiór treningowy (po SMOTE): 228388 rekordów")
    st.write(f"- Zbiór walidacyjny: 54611 rekordów")
    st.write(f"- Zbiór testowy: 54611 rekordów")

elif section == "Analiza Wstępna Kierowców":
    st.title("Analiza Wstępna: Charakterystyka Kierowców w Wypadkach (Wyniki Statyczne)")

    # --- Dane statyczne ---
    total_accidents_static = 273053

    # Tabela 1: Proporcje kierowców
    driver_origin_data = {
        'Pochodzenie': ['Miasto', 'Wieś/Małe Miasto', 'Suma'],
        'Liczba': [222719, 50334, 273053],
        'Procent': [81.6, 18.4, 100.0]
    }
    driver_origin_display = pd.DataFrame(driver_origin_data)

    # Tabela 2: Rozkład wg lat
    driver_stats_data = {
        'Rok': [2021, 2022, 2023],
        'Wieś/Małe Miasto': [15908, 17419, 17007],
        'Procent Wieś/Małe Miasto': [17.8, 18.7, 18.9],
        'Miasto': [73686, 75877, 73156],
        'Procent Miasto': [82.2, 81.3, 81.1],
        'Suma': [89594, 93296, 90163]
    }
    driver_stats_display = pd.DataFrame(driver_stats_data)

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela 1: Proporcje kierowców według miejsca zamieszkania")
    st.dataframe(driver_origin_display.style.format({'Liczba': '{:,.0f}', 'Procent': '{:.1f}%'}))

    st.subheader("Tabela 2: Rozkład kierowców według miejsca zamieszkania w latach 2021-2023")
    st.dataframe(driver_stats_display.style.format({
        'Wieś/Małe Miasto': '{:,.0f}', 'Procent Wieś/Małe Miasto': '{:.1f}%',
        'Miasto': '{:,.0f}', 'Procent Miasto': '{:.1f}%',
        'Suma': '{:,.0f}'
    }))

    st.subheader("Wizualizacje (Odtworzone)")

    # --- Odtworzenie wykresów Matplotlib na podstawie danych statycznych ---
    fig_mpl = plt.figure(figsize=(12, 10))
    gs = fig_mpl.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Wykres 1: Całkowita liczba wypadków
    ax1 = fig_mpl.add_subplot(gs[0, 0])
    bars1 = ax1.bar(['Wszystkie wypadki'], [total_accidents_static], color='#93c47d')
    ax1.set_title('Całkowita liczba analizowanych wypadków')
    ax1.set_ylabel('Liczba')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{int(height):,} (100%)', ha='center', va='center', fontsize=10, color='black')

    # Wykres 2: Proporcje kierowców
    ax2 = fig_mpl.add_subplot(gs[0, 1])
    driver_origin_plot = driver_origin_display[driver_origin_display['Pochodzenie'] != 'Suma'].set_index('Pochodzenie')
    bottom_val = 0
    colors = {'Wieś/Małe Miasto': '#1f77b4', 'Miasto': '#ff7f0e'}
    order = ['Wieś/Małe Miasto', 'Miasto']
    for origin_type in order:
        if origin_type in driver_origin_plot.index:
            value = driver_origin_plot.loc[origin_type, 'Liczba']
            percentage = driver_origin_plot.loc[origin_type, 'Procent']
            bar = ax2.bar(['Kierowcy'], [value], bottom=[bottom_val], color=colors[origin_type], label=origin_type)
            text_y = bottom_val + value / 2
            ax2.text(0, text_y, f"{int(value):,}\n({percentage:.1f}%)", ha='center', va='center', fontsize=10, color='white')
            bottom_val += value

    ax2.set_title('Proporcje kierowców wg miejsca zamieszkania')
    ax2.set_ylabel('Liczba kierowców')
    ax2.set_xticks([])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, total_accidents_static * 1.1)

    # Wykres 3: Rozkład kierowców według lat
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
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height):,}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    autolabel(rects1, ax3)
    autolabel(rects2, ax3)

    fig_mpl.suptitle('Analiza kierowców w wypadkach drogowych (2021-2023)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    st.pyplot(fig_mpl)

elif section == "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku":
    st.title("Analiza Związku: Miejsce Zamieszkania Kierowcy a Lokalizacja Wypadku (Wyniki Statyczne)")

    # --- Dane statyczne ---
    contingency_data = {
        'Wypadek Miejski': [15893, 174431],
        'Wypadek Wiejski': [34441, 48288]
    }
    contingency_table = pd.DataFrame(contingency_data, index=['Wieś/Małe Miasto', 'Miasto'])

    location_stats_data = {
        'Wypadki Miejskie (%)': [31.6, 78.3],
        'Wypadki Wiejskie (%)': [68.4, 21.7]
    }
    location_stats = pd.DataFrame(location_stats_data, index=['Wieś/Małe Miasto', 'Miasto'])

    chi2_stat = 42475.60
    p_value_chi2 = 0.0
    dof_chi2 = 1
    phi_stat = 0.394
    strength = "Umiarkowany (φ = 0.3–0.5)"
    conclusion = f"Odrzucamy hipotezę zerową (H₀). Istnieje statystycznie istotny związek (p < 0.0001)."
    alpha = 0.05

    expected_data = {
        'Wypadek Miejski': [35083.9, 155240.1],
        'Wypadek Wiejski': [15250.1, 67478.9]
    }
    expected_df = pd.DataFrame(expected_data, index=['Wieś/Małe Miasto', 'Miasto'])

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela Kontyngencji (Obserwowane Liczby)")
    st.dataframe(contingency_table.style.format("{:,.0f}"))

    st.subheader("Tabela Procentowa Lokalizacji Wypadków wg Pochodzenia Kierowcy")
    st.dataframe(location_stats.style.format("{:.1f}%"))

    st.subheader("Wyniki Testu Chi-kwadrat Niezależności")
    st.markdown(f"""
    - **Statystyka chi-kwadrat (χ²):** {chi2_stat:.2f}
    - **Wartość p (p-value):** {p_value_chi2:.4e} (bardzo bliska 0)
    - **Stopnie swobody (dof):** {dof_chi2}
    - **Współczynnik Phi (φ):** {phi_stat:.3f}
    - **Interpretacja siły związku (Phi):** {strength}
    - **Wniosek (poziom istotności α = {alpha}):** {conclusion}
    """)

    st.subheader("Tabela Oczekiwana (Gdyby nie było związku)")
    st.dataframe(expected_df.style.format("{:,.1f}"))

    # --- Odtworzenie Wykresu Plotly ---
    st.subheader("Wykres: Procent Wypadków Miejskich i Wiejskich wg Pochodzenia Kierowcy")
    location_stats_plot = location_stats.reset_index().rename(columns={'index': 'Pochodzenie Kierowcy'})
    location_stats_melted = location_stats_plot.melt(
        id_vars='Pochodzenie Kierowcy',
        var_name='Typ Obszaru Wypadku',
        value_name='Procent Wypadków'
    )
    location_stats_melted['Typ Obszaru Wypadku'] = location_stats_melted['Typ Obszaru Wypadku'].str.replace(' (%)', '')

    fig_plotly = px.bar(location_stats_melted,
                         x='Pochodzenie Kierowcy',
                         y='Procent Wypadków',
                         color='Typ Obszaru Wypadku',
                         title='Procent Wypadków Miejskich i Wiejskich<br>wg Miejsca Zamieszkania Kierowcy',
                         labels={'Procent Wypadków': 'Procent Wypadków (%)', 'Typ Obszaru Wypadku': 'Lokalizacja Wypadku'},
                         color_discrete_map={'Wypadki Miejskie': '#1f77b4', 'Wypadki Wiejskie': '#ff7f0e'},
                         barmode='group',
                         text='Procent Wypadków'
                         )
    fig_plotly.update_layout(yaxis_ticksuffix='%', yaxis_title='Procent Wypadków (%)', xaxis_title='Pochodzenie Kierowcy')
    fig_plotly.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- Podsumowanie i wnioski z tej sekcji ---
    st.subheader("Interpretacja i Wnioski z Analizy Związku")
    st.markdown("""
    **Kluczowe obserwacje:**
    - **Kierowcy z obszarów wiejskich/małych miast**: Znacznie częściej uczestniczą w wypadkach na terenach wiejskich (ok. 68.4%) niż miejskich (ok. 31.6%).
    - **Kierowcy z miast**: Dominują w wypadkach na terenach miejskich (ok. 78.3%), a rzadziej uczestniczą w wypadkach na terenach wiejskich (ok. 21.7%).

    **Wyniki testu chi-kwadrat:**
    - Test wykazał **statystycznie istotny związek** (p < 0.0001) między miejscem zamieszkania kierowcy a lokalizacją wypadku.
    - Siła tego związku, mierzona współczynnikiem Phi (φ ≈ 0.394), jest **umiarkowana**. Oznacza to, że miejsce zamieszkania jest ważnym czynnikiem, ale nie jedynym determinującym lokalizację wypadku. Inne czynniki, jak typ drogi, warunki, prędkość, również odgrywają rolę.

    **Wnioski:**
    1. **Istnienie związku**: Potwierdzono wyraźny związek. Kierowcy częściej ulegają wypadkom w środowisku, w którym mieszkają (miejscy w miastach, wiejscy na wsiach), ale dysproporcja jest szczególnie widoczna dla kierowców wiejskich na terenach wiejskich.
    2. **Wstępna weryfikacja hipotezy**: Wyniki **nie potwierdzają** pierwotnej hipotezy, że *kierowcy miejscy* mają *wyższe* prawdopodobieństwo wypadku na wsi. Wręcz przeciwnie, to kierowcy **wiejsko-małomiejscy** mają znacznie wyższy odsetek wypadków na terenach wiejskich w obrębie swojej grupy. Jednak kierowcy miejscy, stanowiąc większość ogółu, nadal generują znaczną liczbę wypadków na wsiach w liczbach bezwzględnych.
    3. **Potrzeba dalszej analizy**: Umiarkowana siła związku sugeruje, że modele ML mogą pomóc zidentyfikować inne czynniki wpływające na ryzyko wypadku w terenie wiejskim, szczególnie interakcje między pochodzeniem kierowcy a innymi zmiennymi.
    """)

elif section == "Opis Modelowania ML":
    st.title("Modelowanie Uczenia Maszynowego - Opis")
    st.header("Cel: Przewidywanie, czy wypadek zdarzy się na terenie wiejskim (`is_rural_accident` = 1)")

    st.subheader("Wybrane Modele:")
    st.markdown("- **XGBoost Classifier:** Wydajny model gradient boostingowy.")
    st.markdown("- **Random Forest Classifier:** Zespół drzew decyzyjnych.")

    st.subheader("Opis Procesu (z oryginalnej analizy):")
    st.markdown("""
    1. Dane zostały podzielone na zbiory: treningowy (60%), walidacyjny (20%) i testowy (20%).
    2. Zastosowano **SMOTE** na zbiorze treningowym, aby zrównoważyć klasy.
    3. Modele zostały wytrenowane na zbalansowanym zbiorze treningowym z użyciem określonych hiperparametrów (przykładowe poniżej).
    4. Ocena modeli odbyła się na **niezmienionych** (niezbalansowanych) zbiorach walidacyjnym i testowym.

    *Ta statyczna wersja aplikacji nie trenuje modeli, jedynie prezentuje wcześniej uzyskane wyniki.*
    """)

    st.subheader("Przykładowe Hiperparametry Użyte w Analizie:")
    st.code("""
# XGBoost
params_xgb = {
    'random_state': 42, 'scale_pos_weight': 1, 'max_depth': 9,
    'n_estimators': 269, 'learning_rate': 0.06, 'reg_alpha': 0.1,
    'reg_lambda': 1.9, 'subsample': 0.8, 'colsample_bytree': 0.6,
}

# RandomForest
params_rf = {
    'random_state': 42, 'n_estimators': 229, 'max_depth': 14,
    'min_samples_split': 54, 'min_samples_leaf': 26, 'n_jobs': -1,
    'max_features': 'sqrt', 'criterion': 'entropy', 'bootstrap': False,
}
    """, language='python')

elif section == "Ocena Modeli":
    st.title("Ocena Modeli Uczenia Maszynowego (Wyniki Statyczne)")
    st.markdown("Ocena przeprowadzona na zbiorach **walidacyjnym** i **testowym** (bez SMOTE). Próg decyzyjny: 0.5.")

    # --- Statyczne Wyniki ---
    report_val_xgb_static = """
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.83      0.79      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
    """
    auc_val_xgb_static = 0.9400

    report_val_rf_static = """
                  precision    recall  f1-score   support

           0       0.92      0.88      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
    """
    auc_val_rf_static = 0.9338

    report_test_xgb_static = """
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.84      0.78      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
    """
    auc_test_xgb_static = 0.9400

    report_test_rf_static = """
                  precision    recall  f1-score   support

           0       0.92      0.89      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
    """
    auc_test_rf_static = 0.9327

    # --- Wyświetlanie wyników ---
    st.subheader("Wyniki na Zbiorze Walidacyjnym")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {auc_val_xgb_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_val_xgb_static)
    with col2:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {auc_val_rf_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_val_rf_static)

    st.subheader("Wyniki na Zbiorze Testowym (Ostateczna Ocena)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {auc_test_xgb_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_test_xgb_static)
    with col4:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {auc_test_rf_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_test_rf_static)

    # --- Krzywe ROC (Odtworzone - przykładowe dane) ---
    st.subheader("Krzywe ROC (Zbiór Testowy - Wykres Ilustracyjny)")
    fpr_xgb_static = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
    tpr_xgb_static = np.array([0, 0.6, 0.8, 0.88, 0.92, 0.96, 1])
    fpr_rf_static = np.array([0, 0.07, 0.15, 0.25, 0.35, 0.55, 1])
    tpr_rf_static = np.array([0, 0.55, 0.75, 0.85, 0.90, 0.94, 1])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_xgb_static, y=tpr_xgb_static, mode='lines', name=f'XGBoost (AUC ≈ {auc_test_xgb_static:.4f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_rf_static, y=tpr_rf_static, mode='lines', name=f'Random Forest (AUC ≈ {auc_test_rf_static:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Losowy Klasyfikator', line=dict(dash='dash')))

    fig_roc.update_layout(
        title='Krzywa ROC - Zbiór Testowy (Ilustracja)',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend_title='Model',
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("Uwaga: Krzywa ROC jest ilustracją opartą na przykładowych danych dla tej wersji statycznej.")

    st.markdown("""
    **Interpretacja wyników:**
    - **AUC-ROC:** Oba modele osiągnęły wysokie wartości AUC (XGBoost 0.9400, RF 0.9327), co wskazuje na bardzo dobrą zdolność do rozróżniania klas. XGBoost jest nieco lepszy.
    - **Precision/Recall/F1-score:** Metryki dla klasy `1` (wypadek wiejski) są kluczowe. XGBoost osiąga lepszy balans (F1=0.81) niż RF (F1=0.79). `Recall` (zdolność wykrywania wypadków wiejskich) jest wysoki dla obu, co jest efektem m.in. zastosowania SMOTE podczas treningu.
    - **Porównanie Walidacja vs Test:** Wyniki są bardzo podobne, co sugeruje dobrą generalizację modeli.
    """)

elif section == "Ważność Cech (XGBoost)":
    st.title("Ważność Cech według Modelu XGBoost (Wyniki Statyczne)")
    st.markdown("Pokazuje, które cechy miały największy wpływ na predykcje modelu XGBoost w oryginalnej analizie.")

    # --- Statyczne Dane Ważności Cech (Top 12) ---
    feature_importance_data = {
        'Cecha': [
            'speed_limit_normalized',
            'urban_driver_speed',
            'is_urban_driver',
            'distance_speed_interaction',
            'junction_detail_1.0',
            'road_type_6',
            'junction_control_4.0',
            'light_conditions_6.0',
            'casualty_type_9.0',
            'important_driver_distance',
            'urban_driver_long_distance',
            'skidding_and_overturning_9.0'
        ],
        'Ważność': [0.1827, 0.1711, 0.0566, 0.0543, 0.0291, 0.0268, 0.0242, 0.0231, 0.0170, 0.0166, 0.0148, 0.0132]
    }
    top_features = pd.DataFrame(feature_importance_data)

    st.subheader("Top 12 najważniejszych cech")
    st.dataframe(top_features.style.format({'Ważność': '{:.4f}'}))

    # Wizualizacja
    st.subheader("Wykres Ważności Cech (Odtworzony)")
    fig_imp = plt.figure(figsize=(10, 8))
    plt.barh(top_features['Cecha'], top_features['Ważność'], color='skyblue')
    plt.xlabel('Ważność (Importance)')
    plt.ylabel('Cecha')
    plt.title('Ważność Cech (XGBoost) - Top 12')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.markdown("""
    **Interpretacja:**
    - Największy wpływ na predykcje miały cechy związane z **prędkością** (`speed_limit_normalized`, `urban_driver_speed`), **pochodzeniem kierowcy** (`is_urban_driver`), **interakcją dystansu i prędkości** (`distance_speed_interaction`), oraz specyficznymi warunkami drogowymi jak **skrzyżowania typu Y** (`junction_detail_1.0`), **drogi jednopasmowe** (`road_type_6`), **brak kontroli ruchu** (`junction_control_4.0`), i **ciemność bez oświetlenia** (`light_conditions_6.0`).
    """)

elif section == "Analiza Kluczowych Cech (Chi-kwadrat)":
    st.title("Szczegółowa Analiza Kluczowych Cech vs Lokalizacja Wypadku (Test Chi-kwadrat - Wyniki Statyczne)")

    # --- Statyczne wyniki testów Chi-kwadrat ---
    chi2_results_data = {
        'Cecha': [
            'is_urban_driver', 'road_type', 'junction_control', 'junction_detail',
            'important_driver_distance', 'light_conditions', 'casualty_type'
        ],
        'chi2': [42475.6, 349.1, 9180.7, 2669.3, 13160.5, 13593.7, 8562.5],
        'p_value': [0.0, 1.543e-76, 0.0, 0.0, 0.0, 0.0, 0.0],
        'V': [0.394, 0.036, 0.183, 0.099, 0.220, 0.223, 0.177],
        'Interpretacja': ['Umiarkowany związek', 'Słaby związek', 'Umiarkowany związek', 'Słaby związek', 'Umiarkowany związek', 'Umiarkowany związek', 'Umiarkowany związek']
    }
    results_df = pd.DataFrame(chi2_results_data).set_index('Cecha')

    st.subheader("Wyniki Testów Chi-kwadrat dla Kluczowych Cech")
    st.dataframe(results_df.style.format({
        'chi2': '{:.1f}',
        'p_value': '{:.1e}',
        'V': '{:.3f}'
    }))

    st.markdown("""
    **Interpretacja:**
    - Testy chi-kwadrat wykazały statystycznie istotny związek (p < 0.05) dla wszystkich badanych cech.
    - Najsilniejszy związek z lokalizacją wypadku ma **is_urban_driver** (V=0.394), co potwierdza jego kluczową rolę.
    - Cechy takie jak **important_driver_distance**, **light_conditions**, i **junction_control** również wykazują umiarkowany wpływ.
    - **road_type** i **junction_detail** mają słabszy, ale nadal istotny statystycznie związek.
    """)

elif section == "Wnioski i Podsumowanie":
    st.title("Wnioski Końcowe i Podsumowanie Analizy")

    st.header("Podsumowanie Wyników")
    st.markdown("""
    1. **Związek Miejsca Zamieszkania z Lokalizacją Wypadku:**
        - Potwierdzono **statystycznie istotny, umiarkowany związek** (φ ≈ 0.394).
        - Kierowcy z obszarów **wiejskich/małomiejskich** częściej uczestniczą w wypadkach na terenach **wiejskich** (ok. 68%) niż kierowcy miejscy (ok. 22%).
        - Kierowcy **miejscy** dominują w wypadkach **miejskich** (ok. 78%).
        - Wyniki **nie potwierdzają** pierwotnej hipotezy o *wyższym* ryzyku kierowców *miejskich* na terenach *wiejskich*.

    2. **Modelowanie Predykcyjne:**
        - Modele ML (XGBoost, RF) wykazały **bardzo dobrą zdolność** (AUC ≈ 0.94) do przewidywania lokalizacji wypadku (wiejska vs miejska).
        - Modele dobrze generalizowały wyniki na danych testowych.

    3. **Kluczowe Czynniki Ryzyka (Ważność Cech):**
        - Najważniejsze okazały się cechy związane z **prędkością** (`speed_limit_normalized`, `urban_driver_speed`), **pochodzeniem kierowcy** (`is_urban_driver`), **interakcją dystansu i prędkości** (`distance_speed_interaction`), oraz specyficznymi warunkami drogowymi jak **skrzyżowania typu Y** (`junction_detail_1.0`), **drogi jednopasmowe** (`road_type_6`), **brak kontroli ruchu** (`junction_control_4.0`), i **ciemność bez oświetlenia** (`light_conditions_6.0`).

    4. **Odpowiedzi na Pytania Badawcze:**
        - **Miejsce zamieszkania ma wpływ?** Tak, istotny statystycznie i praktycznie.
        - **Jakie cechy determinują ryzyko?** Kombinacja cech prędkości, pochodzenia kierowcy, dystansu oraz specyficznych warunków drogowych.
        - **Czy modele potwierdzają hipotezę?** Nie, ale pokazują złożoność interakcji.
    """)

    st.header("Ograniczenia Analizy")
    st.markdown("""
    - Dane `STATS19` obejmują tylko zgłoszone wypadki z obrażeniami.
    - Jakość danych (np. dokładność lokalizacji) może wpływać na wyniki.
    - Analiza nie uwzględniała natężenia ruchu ani dokładnych tras.
    - Korelacja nie implikuje przyczynowości.
    """)

    st.header("Rekomendacje i Dalsze Kierunki Badań")
    st.markdown("""
    - Skupienie prewencji na specyficznych typach dróg i skrzyżowań wiejskich.
    - Analiza wpływu doświadczenia kierowcy.
    - Badanie interakcji między miejscem zamieszkania a innymi czynnikami.
    - Wykorzystanie bardziej zaawansowanych technik modelowania (jeśli dostępne dane).
    """)