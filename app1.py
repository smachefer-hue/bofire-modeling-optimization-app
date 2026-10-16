import streamlit as st
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# BoFire Core Imports
from bofire.data_models.domain.api import Domain, Inputs, Outputs, Constraints
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.constraints.api import LinearEqualityConstraint
import bofire.surrogates.api as surrogates
import bofire.strategies.api as strategies
from bofire.data_models.surrogates.api import (
    RandomForestSurrogate, 
    SingleTaskGPSurrogate,
    LinearSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate
)

# Warnungen unterdrücken
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*train_inputs.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*not writable.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script.*")

st.set_page_config(layout="wide")
st.title("📊 BoFire Optimierungs & Metamodellierungs Pipeline")
st.markdown("Laden Sie Ihre CSV-Datei hoch, um die Korrelationsanalyse, Modellierung und Bayes'sche Optimierung zu starten.")

# 1. Datei-Upload
uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei (*.csv)", type="csv")

if uploaded_file is not None:
    # Daten einlesen
    data = pd.read_csv(uploaded_file, sep=";")
    data.columns = data.columns.str.strip()
    data = data.apply(pd.to_numeric, errors='coerce').copy()
    
    st.subheader("Datenvorschau")
    st.dataframe(data.head())
    
    # Automatische Variablen-Erkennung
    input_keys = [col for col in data.columns if col.startswith('I')]
    output_keys = [col for col in data.columns if col.startswith('O')]
    
    st.info(f"**Erkannte Eingangsvariablen (Inputs):** {', '.join(input_keys)}")
    st.info(f"**Erkannte Ausgangsvariablen (Outputs):** {', '.join(output_keys)}")
    
    if not input_keys or not output_keys:
        st.error("Fehler: Es konnten keine Spalten gefunden werden, die mit 'I' (Inputs) oder 'O' (Outputs) beginnen!")
    else:
        # Korrelationsmatrix
        st.subheader("📈 Erweiterte Korrelationsmatrix")
        korrelations_spalten = input_keys + output_keys
        aktive_spalten = [col for col in korrelations_spalten if data[col].nunique() > 1]
        
        if len(aktive_spalten) > 1:
            def plot_correlation_coefficients(x, y, **kwargs):
                r = x.corr(y)
                ax = plt.gca()
                if np.isnan(r): r = 0.0
                cmap = plt.get_cmap("coolwarm")
                normalized_r = (r + 1) / 2  
                ax.set_facecolor(cmap(normalized_r))
                text_color = "white" if abs(r) > 0.5 else "black"
                ax.annotate(f"{r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes, fontsize=10, fontweight="bold", ha="center", va="center", color=text_color)
                ax.set_xticks([])
                ax.set_yticks([])

            fig = plt.figure(figsize=(10, 10))
            grid = sns.PairGrid(data, vars=aktive_spalten, height=1.5)
            grid.map_lower(sns.scatterplot, alpha=0.6, color="purple", edgecolor="k", s=25)
            grid.map_diag(sns.histplot, kde=True, color="gray", alpha=0.6)
            grid.map_upper(plot_correlation_coefficients)
            plt.subplots_adjust(top=0.93)
            grid.fig.suptitle('Erweiterte Korrelationsmatrix', fontsize=14, fontweight='bold')
            st.pyplot(grid.fig)
        
        # Konfiguration des Mischungs-Constraints
        st.subheader("🛠️ Konfiguration des Mischungs-Constraints & Modell-Auswahl")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            default_gemisch = [k for k in input_keys if "_" in k]
            gewaehlte_komponenten = st.multiselect("Mischungskomponenten auswählen:", options=input_keys, default=default_gemisch)
        with col2:
            ziel_summe = st.number_input("Ziel-Summe des Gemischs:", value=1.0, step=0.1)
            GEWAEHLTES_MODELL = st.selectbox("Wählen Sie den Metamodell-Typ:", options=["linear", "random_forest", "standard_gp", "saas_gp", "mixed_gp"])
        with col3:
            n_splits = st.number_input("Anzahl Faltungen für Kreuzvalidierung (K-Fold):", min_value=2, max_value=20, value=5, step=1)
        
        if st.button("Modellierung & Analyse starten", key="start_pipeline"):
            # Domain-Erstellung
            input_features = []
            for key in input_keys:
                min_val = float(data[key].min())
                max_val = float(data[key].max())
                if min_val == max_val: max_val += 0.001
                input_features.append(ContinuousInput(key=key, bounds=(min_val, max_val)))
                
            output_features = [ContinuousOutput(key=key, objective=MaximizeObjective()) for key in output_keys]
            
            constraints_list = []
            if len(gewaehlte_komponenten) > 1:
                summen_constraint = LinearEqualityConstraint(
                    features=gewaehlte_komponenten,
                    coefficients=[1.0] * len(gewaehlte_komponenten),
                    rhs=ziel_summe
                )
                constraints_list.append(summen_constraint)
                
            domain = Domain(inputs=Inputs(features=input_features), outputs=Outputs(features=output_features), constraints=Constraints(constraints=constraints_list))
            
            # Hilfsfunktion zur Instanziierung des gewählten Modells
            def get_model_data_model(model_type, domain_obj):
                if model_type == "random_forest":
                    return RandomForestSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs, n_estimators=100)
                elif model_type == "standard_gp":
                    return SingleTaskGPSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs)
                elif model_type == "linear":
                    return LinearSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs)
                elif model_type == "saas_gp":
                    return EnsembleMapSaasSingleTaskGPSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs)
                elif model_type == "mixed_gp":
                    return MixedSingleTaskGPSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs)
                return LinearSurrogate(inputs=domain_obj.inputs, outputs=domain_obj.outputs)

            primary_output_key = output_keys[0]
            
            # --- START K-FOLD KREUZVALIDIERUNG & Q² BERECHNUNG ---
            st.subheader(f"🔄 {n_splits}-Fold Kreuzvalidierung läuft...")
            
            kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=42)
            cv_predictions = np.zeros(len(data))
            
            progress_bar = st.progress(0.0)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
                train_data = data.iloc[train_idx].reset_index(drop=True)
                test_data = data.iloc[test_idx].reset_index(drop=True)
                
                # Lokales Modell für den aktuellen Fold erzeugen und trainieren
                fold_data_model = get_model_data_model(GEWAEHLTES_MODELL, domain)
                fold_model = surrogates.map(fold_data_model)
                fold_model.fit(train_data)
                
                # Vorhersage für den Test-Split
                fold_pred = fold_model.predict(test_data)
                cv_predictions[test_idx] = fold_pred[f"{primary_output_key}_pred"].values
                
                progress_bar.progress((fold + 1) / n_splits)
            
            progress_bar.empty()
            
            # Berechnung des Prognosekoeffizienten Q²
            actual_values = data[primary_output_key].values
            mean_actual = np.mean(actual_values)
            
            ss_res = np.sum((actual_values - cv_predictions) ** 2)
            ss_tot = np.sum((actual_values - mean_actual) ** 2)
            q2_prognose = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            rmse_cv = np.sqrt(mean_squared_error(actual_values, cv_predictions))
            # --- ENDE KREUZVALIDIERUNG ---

            # Vollständiges Modell auf allen Daten trainieren (für die Plots und BO)
            model_data_model = get_model_data_model(GEWAEHLTES_MODELL, domain)
            model = surrogates.map(model_data_model)
            model.fit(data)
            st.success(f"Modell ({GEWAEHLTES_MODELL.upper()}) erfolgreich trainiert und kreuzvalidiert!")
            
            # Diagnose-Plot (Ergänzt um CV-Daten)
            st.subheader("🎯 Modell-Diagnose: Predicted vs. Actual")
            train_predicted = model.predict(data)[f"{primary_output_key}_pred"].values
            r2_train = r2_score(actual_values, train_predicted)
            
            fig_diag, ax_diag = plt.subplots(figsize=(7, 5))
            ax_diag.scatter(actual_values, train_predicted, color='purple', alpha=0.5, edgecolors='k', s=40, label='Trainingsdaten (Fit)')
            ax_diag.scatter(actual_values, cv_predictions, color='orange', alpha=0.7, marker='x', s=40, label=f'Kreuzvalidierung (Test)')
            
            min_v = min(min(actual_values), min(train_predicted), min(cv_predictions))
            max_v = max(max(actual_values), max(train_predicted), max(cv_predictions))
            ax_diag.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=2, label='Ideallinie ($y=x$)')
            
            ax_diag.set_title(f'Predicted vs. Actual ({GEWAEHLTES_MODELL.upper()})')
            ax_diag.set_xlabel(f'Reale Messwerte ({primary_output_key})')
            ax_diag.set_ylabel('Modell-Vorhersage')
            
            info_text = (
                f"Modell: {GEWAEHLTES_MODELL.upper()}\n"
                f"Train-$R^2$: {r2_train:.3f}\n"
                f"Prognose-$Q^2$ (CV): {q2_prognose:.3f}\n"
                f"RMSE (CV): ± {rmse_cv:.3f}"
            )
            ax_diag.text(0.05, 0.95, info_text, transform=ax_diag.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
            ax_diag.legend(loc='lower right')
            ax_diag.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_diag)
            
            # Metriken übersichtlich anzeigen
            m1, m2, m3 = st.columns(3)
            m1.metric("Bestimmtheitsmaß ($R^2$ Train)", f"{r2_train:.3f}")
            m2.metric("Prognosekoeffizient ($Q^2$ CV)", f"{q2_prognose:.3f}", delta=f"{q2_prognose - r2_train:.3f} vs Train", delta_color="inverse")
            m3.metric("Wurzel der mittleren Fehlerquadratsumme (RMSE CV)", f"{rmse_cv:.3f}")
            
            # Speicher für die interaktive App bereitstellen
            st.session_state['domain'] = domain
            st.session_state['model'] = model
            st.session_state['gewaehlte_komponenten'] = gewaehlte_komponenten
            st.session_state['ziel_summe'] = ziel_summe
            st.session_state['primary_output_key'] = primary_output_key
            st.session_state['input_keys'] = input_keys
            st.session_state['data'] = data

# Interaktiver 3D Plot-Bereich außerhalb des Main-Buttons
if 'model' in st.session_state:
    st.subheader("🔮 Interaktive 3D-Oberflächenanalyse")
    
    domain = st.session_state['domain']
    model = st.session_state['model']
    gewaehlte_komponenten = st.session_state['gewaehlte_komponenten']
    ziel_summe = st.session_state['ziel_summe']
    primary_output_key = st.session_state['primary_output_key']
    input_keys = st.session_state['input_keys']
    data = st.session_state['data']
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        x_axis = st.selectbox("X-Achse:", options=input_keys, index=0)
        y_axis = st.selectbox("Y-Achse:", options=input_keys, index=min(1, len(input_keys)-1))
    with col_ctrl2:
        farbskala = st.selectbox("Farbskala:", options=['Plasma', 'Viridis', 'Cividis', 'Inferno', 'Hot'])
        slider_opacity = st.slider("Flächen-Transparenz:", 0.1, 1.0, 0.7, 0.05)
        
    st.markdown("##### Werte der übrigen (festen) Variablen einstellen:")
    fixed_sliders = {}
    for key in input_keys:
        if key not in [x_axis, y_axis]:
            feat = domain.inputs.get_by_key(key)
            mean_val = (feat.bounds[0] + feat.bounds[1]) / 2.0
            fixed_sliders[key] = st.slider(f"Wert für {key}:", float(feat.bounds[0]), float(feat.bounds[1]), float(mean_val))

    grid_size = 30
    x_feat = domain.inputs.get_by_key(x_axis)
    y_feat = domain.inputs.get_by_key(y_axis)
    x_range = np.linspace(x_feat.bounds[0], x_feat.bounds[1], grid_size)
    y_range = np.linspace(y_feat.bounds[0], y_feat.bounds[1], grid_size)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    
    plot_df_list = []
    for x_val in x_range:
        for y_val in y_range:
            row_data = {k: val for k, val in fixed_sliders.items()}
            row_data[x_axis] = x_val
            row_data[y_axis] = y_val
            plot_df_list.append(row_data)
            
    plot_df = pd.DataFrame(plot_df_list)
    predictions_grid = model.predict(plot_df)
    Z_mesh = predictions_grid[f"{primary_output_key}_pred"].values.reshape(grid_size, grid_size).T
    
    valid_x, valid_y, valid_z = [], [], []
    line_grid = 100
    x_line_range = np.linspace(x_feat.bounds[0], x_feat.bounds[1], line_grid)
    y_line_range = np.linspace(y_feat.bounds[0], y_feat.bounds[1], grid_size)
    
    for x_val in x_line_range:
        for y_val in y_line_range:
            current_mix = {k: fixed_sliders.get(k, 0.0) for k in gewaehlte_komponenten}
            if x_axis in gewaehlte_komponenten: current_mix[x_axis] = x_val
            if y_axis in gewaehlte_komponenten: current_mix[y_axis] = y_val
            if abs(sum(current_mix.values()) - ziel_summe) <= (0.01 * ziel_summe):
                valid_x.append(x_val)
                valid_y.append(y_val)
                
    if len(valid_x) > 0:
        line_df_list = []
        for i in range(len(valid_x)):
            row_data = {k: fixed_sliders.get(k, 0.0) for k in input_keys}
            row_data[x_axis] = valid_x[i]
            row_data[y_axis] = valid_y[i]
            line_df_list.append(row_data)
        line_df = pd.DataFrame(line_df_list)
        line_predictions = model.predict(line_df)
        valid_z = line_predictions[f"{primary_output_key}_pred"].values
        
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X_mesh, y=Y_mesh, z=Z_mesh, colorscale=farbskala, opacity=slider_opacity, showscale=True))
    
    if len(valid_z) > 0:
        fig.add_trace(go.Scatter3d(x=valid_x, y=valid_y, z=valid_z, mode='lines', line=dict(color='black', width=8), name='Erlaubter Pfad'))
        
    fig.update_layout(scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=primary_output_key), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)
    
    # Bayes'sche Optimierung
    st.subheader("🤖 Bayes'sche Optimierung (SPO-Strategie)")
    if st.button("Optimale nächste Experimente berechnen"):
        try:
            from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
            from bofire.data_models.acquisition_functions.api import qLogEI
            neue_inputs = []
            fixed_values_for_output = {}
            entfernte_features = []
            
            for key in input_keys:
                hist_min = float(data[key].min())
                hist_max = float(data[key].max())
                if abs(hist_max - hist_min) < 1e-6:
                    fixed_values_for_output[key] = hist_min
                    entfernte_features.append(key)
                else:
                    feat = domain.inputs.get_by_key(key)
                    feat.bounds = (hist_min, hist_max)
                    neue_inputs.append(feat)
                    
            neue_constraints = []
            for c in domain.constraints.get():
                if hasattr(c, 'features'):
                    aktive_c_features = [f for f in c.features if f not in entfernte_features]
                    entfernte_c_features = [f for f in c.features if f in entfernte_features]
                    abzug = sum([fixed_values_for_output[f] for f in entfernte_c_features])
                    if len(aktive_c_features) > 0:
                        neue_constraints.append(LinearEqualityConstraint(features=aktive_c_features, coefficients=[1.0] * len(aktive_c_features), rhs=c.rhs - abzug))
                        
            optimierungs_domain = Domain(inputs=Inputs(features=neue_inputs), outputs=domain.outputs, constraints=Constraints(constraints=neue_constraints))
            strategy_data_model = SoboStrategyDataModel(domain=optimierungs_domain, acquisition_function=qLogEI())
            strategy = strategies.map(strategy_data_model)
            strategy.num_restarts = 2
            strategy.raw_samples = 32
            strategy.tell(experiments=data)
            proposals = strategy.ask(candidate_count=3)
            
            for key, value in fixed_values_for_output.items():
                proposals[key] = value
            if len(gewaehlte_komponenten) > 1:
                proposals["Kontrollsumme_Gemisch"] = proposals[gewaehlte_komponenten].sum(axis=1)
                
            st.write("Generierte Vorschläge für die nächsten Experimente:")
            st.dataframe(proposals)
            csv_export = proposals.to_csv(sep=";", index=False).encode('utf-8')
            st.download_button(
                label="📥 Nächste Experimente als CSV herunterladen",
                data=csv_export,
                file_name="BoFire_Naechste_Experimente.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"Optimierungshinweis: {str(e)}")
