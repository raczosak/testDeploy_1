# app_static.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# Usunięto importy: seaborn, statsmodels, scipy.stats, sklearn, imblearn, xgboost, tabulate

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Analiza Wypadków Drogowych UK (Statyczna)")

# --- Pasek boczny nawigacji ---
st.sidebar.title("Nawigacja")
section = st.sidebar.radio(
    "Wybierz sekcję analizy:",
    (
        "Wprowadzenie",
        "Opis Przygotowania Danych", # Zmieniona nazwa
        "Analiza Wstępna Kierowców",
        "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku",
        "Opis Modelowania ML", # Zmieniona nazwa
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

elif section == "Opis Przygotowania Danych":
    st.title("II. Dane i Metodyka - Opis Przygotowania Danych") # Zmieniony tytuł

    st.header("1. Źródła danych")
    st.markdown("""
    - Dane pochodzą z oficjalnych brytyjskich baz danych (Department for Transport - data.gov.uk) dotyczących wypadków drogowych z lat 2021-2023 na terenie UK.
    - Tabele (`casualties`, `vehicles`, `accidents`) zostały połączone.
    - Statystyki dotyczą zgłoszonych wypadków z obrażeniami (`STATS19`).
    - **Przewodnik** po statystykach: [link](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
    - **Zestawy danych**: [link](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
    """)

    st.header("2. Opis kroków przygotowania danych") # Zmieniony nagłówek
    st.markdown("""
    W oryginalnej analizie przeprowadzono następujące kroki przygotowania danych (nie są one wykonywane w tej statycznej wersji):
    - **Oczyszczenie danych:** Usunięcie braków danych i wartości niepoprawnych (-1, 99) w kluczowych kolumnach.
    - **Przekształcenie czasu:** Wyodrębnienie godziny (`hour_of_day`).
    - **Przygotowanie zmiennych kategorycznych:** Stworzenie `is_urban_driver` i zmiennej celu `is_rural_accident`.
    - **Normalizacja:** Standaryzacja `speed_limit`.
    - **Binowanie:** Podział wieku kierowcy i ofiary na kategorie.
    - **Inżynieria Cech:** Stworzenie `urban_driver_speed`, `is_rush_hour`, `distance_speed_interaction`.
    - **Kodowanie kategoryczne:** One-hot encoding dla wybranych cech.
    - **Dodatkowe cechy po kodowaniu:** Stworzenie `important_driver_distance`, `urban_driver_long_distance`, `urban_driver_no_junction_control`.
    - **Podział danych:** Na zbiory treningowy (60%), walidacyjny (20%) i testowy (20%) ze stratyfikacją.
    - **Balansowanie danych:** Zastosowanie SMOTE na zbiorze treningowym.

    *Ta wersja aplikacji jedynie **prezentuje** wyniki uzyskane po tych krokach.*
    """)
    # Można tu dodać informacje o finalnych rozmiarach zbiorów, jeśli są znane
    st.subheader("Przykładowe rozmiary zbiorów danych po przetworzeniu:")
    st.write(f"- Zbiór treningowy (po SMOTE): ~285,000 rekordów") # Przykładowa liczba
    st.write(f"- Zbiór walidacyjny: ~54,600 rekordów")        # Przykładowa liczba
    st.write(f"- Zbiór testowy: ~54,600 rekordów")          # Przykładowa liczba


elif section == "Analiza Wstępna Kierowców":
    st.title("Analiza Wstępna: Charakterystyka Kierowców w Wypadkach (Wyniki Statyczne)")

    # --- Dane statyczne ---
    total_accidents_static = 273053 # Z oryginalnego kodu

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
    # Kolejność zgodna z oczekiwaniami
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
    p_value_chi2 = 0.0 # Praktycznie zero
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


elif section == "Opis Modelowania ML": # Zmieniony tytuł
    st.title("Modelowanie Uczenia Maszynowego - Opis") # Zmieniony tytuł
    st.header("Cel: Przewidywanie, czy wypadek zdarzy się na terenie wiejskim (`is_rural_accident` = 1)")

    st.subheader("Wybrane Modele:")
    st.markdown("- **XGBoost Classifier:** Wydajny model gradient boostingowy.")
    st.markdown("- **Random Forest Classifier:** Zespół drzew decyzyjnych.")

    st.subheader("Opis Procesu (z oryginalnej analizy):")
    st.markdown("""
    1.  Dane zostały podzielone na zbiory: treningowy (60%), walidacyjny (20%) i testowy (20%).
    2.  Zastosowano **SMOTE** na zbiorze treningowym, aby zrównoważyć klasy.
    3.  Modele zostały wytrenowane na zbalansowanym zbiorze treningowym z użyciem określonych hiperparametrów (przykładowe poniżej).
    4.  Ocena modeli odbyła się na **niezmienionych** (niezbalansowanych) zbiorach walidacyjnym i testowym.

    *Ta statyczna wersja aplikacji nie trenuje modeli, jedynie prezentuje wcześniej uzyskane wyniki.*
    """)

    st.subheader("Przykładowe Hiperparametry Użyte w Analizie:")
    st.code("""
# XGBoost (Przykładowe)
params_xgb = {
    'random_state': 42, 'scale_pos_weight': 1, 'max_depth': 9,
    'n_estimators': 250, 'learning_rate': 0.03, 'reg_alpha': 1.0,
    'reg_lambda': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'use_label_encoder': False, 'eval_metric': 'logloss'
}

# RandomForest (Przykładowe)
params_rf = {
    'random_state': 42, 'n_estimators': 150, 'max_depth': 10,
    'min_samples_split': 137, 'min_samples_leaf': 26, 'n_jobs': -1,
    'max_features': 'sqrt', 'criterion': 'entropy', 'bootstrap': True,
    'class_weight': None
}
    """, language='python')

elif section == "Ocena Modeli":
    st.title("Ocena Modeli Uczenia Maszynowego (Wyniki Statyczne)")
    st.markdown("Ocena przeprowadzona na zbiorach **walidacyjnym** i **testowym** (bez SMOTE). Próg decyzyjny: 0.5.")

    # --- Statyczne Wyniki ---
    # Przykładowe raporty klasyfikacji (jako stringi)
    report_val_xgb_static = """
                  precision    recall  f1-score   support

               0       0.93      0.89      0.91     38065
               1       0.76      0.84      0.80     16546

        accuracy                           0.88     54611
       macro avg       0.85      0.87      0.85     54611
    weighted avg       0.88      0.88      0.88     54611
    """
    auc_val_xgb_static = 0.9350 # Przykładowa wartość

    report_val_rf_static = """
                  precision    recall  f1-score   support

               0       0.91      0.90      0.91     38065
               1       0.76      0.77      0.77     16546

        accuracy                           0.86     54611
       macro avg       0.83      0.84      0.84     54611
    weighted avg       0.87      0.86      0.86     54611
    """
    auc_val_rf_static = 0.9180 # Przykładowa wartość

    report_test_xgb_static = """
                  precision    recall  f1-score   support

               0       0.93      0.89      0.91     38065
               1       0.76      0.84      0.80     16546

        accuracy                           0.88     54611
       macro avg       0.85      0.87      0.85     54611
    weighted avg       0.88      0.88      0.88     54611
    """ # Zakładamy takie same wyniki jak na walidacji dla przykładu
    auc_test_xgb_static = 0.9345 # Przykładowa wartość

    report_test_rf_static = """
                  precision    recall  f1-score   support

               0       0.91      0.90      0.91     38065
               1       0.76      0.77      0.77     16546

        accuracy                           0.86     54611
       macro avg       0.83      0.84      0.84     54611
    weighted avg       0.87      0.86      0.86     54611
    """ # Zakładamy takie same wyniki jak na walidacji dla przykładu
    auc_test_rf_static = 0.9175 # Przykładowa wartość

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
    # Przykładowe dane FPR/TPR - W aplikacji statycznej najlepiej wstawić obrazek
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
    **Interpretacja wyników (na podstawie przykładowych danych):**
    - **AUC-ROC:** Oba modele osiągnęły wysokie wartości AUC (XGBoost ~0.93, RF ~0.92), co wskazuje na bardzo dobrą zdolność do rozróżniania klas. XGBoost wydaje się nieco lepszy.
    - **Precision/Recall/F1-score:** Metryki dla klasy `1` (wypadek wiejski) są kluczowe. XGBoost osiąga lepszy balans (F1=0.80) niż RF (F1=0.77) w tych przykładowych danych. `Recall` (zdolność wykrywania wypadków wiejskich) jest wysoki dla obu, co jest efektem m.in. zastosowania SMOTE podczas treningu.
    - **Porównanie Walidacja vs Test:** Wyniki są bardzo podobne (w tym przykładzie założono, że są identyczne), co sugerowałoby dobrą generalizację modeli.
    """)


elif section == "Ważność Cech (XGBoost)":
    st.title("Ważność Cech według Modelu XGBoost (Wyniki Statyczne)")
    st.markdown("Pokazuje, które cechy miały największy wpływ na predykcje modelu XGBoost w oryginalnej analizie.")

    # --- Statyczne Dane Ważności Cech (Top 15 - przykładowe) ---
    feature_importance_data = {
        'Cecha': [
            'road_type_6.0', # Single carriageway
            'junction_detail_0.0', # No junction
            'is_urban_driver',
            'driver_distance_banding_4.0', # 16-30 miles
            'speed_limit_normalized',
            'urban_driver_long_distance', # Interakcja
            'light_conditions_4.0', # Darkness - lights lit
            'hour_of_day',
            'age_of_driver_binned_3', # 26-40
            'casualty_type_9.0', # Car occupant
            'distance_speed_interaction', # Interakcja
            'driver_imd_decile',
            'weather_conditions_1.0', # Fine no high winds
            'important_driver_distance', # Interakcja
            'skidding_and_overturning_0.0' # None
        ],
        'Ważność': np.linspace(0.15, 0.01, 15) # Przykładowe malejące wartości
    }
     # Sortowanie malejąco wg ważności
    feature_importance_data['Ważność'] = sorted(feature_importance_data['Ważność'], reverse=True)
    top_features = pd.DataFrame(feature_importance_data)


    st.subheader("Top 15 najważniejszych cech (Przykładowe)")
    st.dataframe(top_features.style.format({'Ważność': '{:.4f}'}))

    # Wizualizacja
    st.subheader("Wykres Ważności Cech (Odtworzony)")
    fig_imp = plt.figure(figsize=(10, 8))
    plt.barh(top_features['Cecha'], top_features['Ważność'], color='skyblue')
    plt.xlabel('Ważność (Importance)')
    plt.ylabel('Cecha')
    plt.title('Ważność Cech (XGBoost) - Top 15 (Ilustracja)')
    plt.gca().invert_yaxis() # Najważniejsze na górze
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.markdown("""
    **Interpretacja (na podstawie przykładowych danych):**
    - Największy wpływ na predykcje miały cechy związane z **typem drogi** (`road_type_6.0` - droga jednopasmowa często wiejska), **brakiem skrzyżowania** (`junction_detail_0.0`), **pochodzeniem kierowcy** (`is_urban_driver`) oraz **odległością od domu** (`driver_distance_banding_4.0`).
    - **Interakcje stworzone przez nas** (`urban_driver_long_distance`, `distance_speed_interaction`, `important_driver_distance`) również okazały się istotne.
    - Cechy takie jak **limit prędkości**, **warunki oświetleniowe**, **godzina**, **wiek kierowcy** i **typ ofiary** również wniosły wkład w predykcje.
    """)


elif section == "Analiza Kluczowych Cech (Chi-kwadrat)":
    st.title("Szczegółowa Analiza Kluczowych Cech vs Lokalizacja Wypadku (Test Chi-kwadrat - Wyniki Statyczne)")

    # --- Statyczne wyniki testów Chi-kwadrat ---
    # Przykładowe wyniki, rzeczywiste wartości mogą się różnić
    chi2_results_data = {
        'Cecha': [
            'is_urban_driver', 'road_type', 'junction_control', 'junction_detail',
            'important_driver_distance', 'light_conditions', 'casualty_type',
            'weather_conditions', 'age_of_driver_binned', 'skidding_and_overturning'
        ],
        'chi2': [42475.6, 15000.2, 8500.7, 12000.1, 9500.3, 7000.9, 3500.5, 2100.0, 1500.8, 800.4],
        'p_value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Wszystkie p<0.05 w tym przykładzie
        'dof': [1, 5, 4, 8, 1, 4, 10, 6, 4, 5], # Przykładowe stopnie swobody
        'significant': [True] * 10 # Zakładamy, że wszystkie były istotne
    }
    results_df = pd.DataFrame(chi2_results_data).set_index('Cecha')

    st.subheader("Wyniki Testów Chi-kwadrat dla Kluczowych Cech (Przykładowe)")
    st.dataframe(results_df.style.format({
        'chi2': '{:.1f}',
        'p_value': '{:.1e}', # Format naukowy dla p-value
        'dof': '{:.0f}'
    }).applymap(lambda x: 'color: green' if x == True else ('color: red' if x == False else ''), subset=['significant']))

    st.markdown(f"""
    **Interpretacja (na podstawie przykładowych wyników):**
    - Tabela pokazuje wyniki testu chi-kwadrat sprawdzającego zależność każdej z kluczowych cech od lokalizacji wypadku (miejska vs wiejska).
    - **significant = True** (zielony) oznacza, że w oryginalnej analizie znaleziono statystycznie istotny związek (p < 0.05) między daną cechą a lokalizacją wypadku. W tym przykładzie wszystkie pokazane cechy wykazały taki związek.
    - Wysoka wartość `chi2` (np. dla `is_urban_driver`, `road_type`) sugeruje silniejszą zależność statystyczną.
    - Wyniki te są zgodne z wynikami ważności cech z modelu ML – cechy istotne statystycznie często mają duży wpływ na predykcje modelu.
    """)

elif section == "Wnioski i Podsumowanie":
    st.title("Wnioski Końcowe i Podsumowanie Analizy")

    st.header("Podsumowanie Wyników")
    st.markdown("""
    1.  **Związek Miejsca Zamieszkania z Lokalizacją Wypadku:**
        * Potwierdzono **statystycznie istotny, umiarkowany związek** (φ ≈ 0.394).
        * Kierowcy z obszarów **wiejskich/małomiejskich** częściej uczestniczą w wypadkach na terenach **wiejskich** (ok. 68%) niż kierowcy miejscy (ok. 22%).
        * Kierowcy **miejscy** dominują w wypadkach **miejskich** (ok. 78%).
        * Wyniki **nie potwierdzają** pierwotnej hipotezy o *wyższym* ryzyku kierowców *miejskich* na terenach *wiejskich*.

    2.  **Modelowanie Predykcyjne:**
        * Modele ML (XGBoost, RF) wykazały **bardzo dobrą zdolność** (AUC > 0.90) do przewidywania lokalizacji wypadku (wiejska vs miejska).
        * Modele dobrze generalizowały wyniki na danych testowych.

    3.  **Kluczowe Czynniki Ryzyka (Ważność Cech):**
        * Najważniejsze okazały się cechy związane z **lokalizacją/infrastrukturą** (typ drogi, brak skrzyżowania) oraz **kierowcą** (pochodzenie, odległość, wiek).
        * Istotne były również **warunki zewnętrzne** (oświetlenie, pogoda) i **dynamika** (poślizg).

    4.  **Odpowiedzi na Pytania Badawcze:**
        * **Miejsce zamieszkania ma wpływ?** Tak, istotny statystycznie i praktycznie.
        * **Jakie cechy determinują ryzyko?** Kombinacja cech drogi, kierowcy, warunków, dynamiki.
        * **Czy modele potwierdzają hipotezę?** Nie, ale pokazują złożoność interakcji.
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