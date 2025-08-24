#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema di Raccomandazione per Persone con Disabilità
Interfaccia Streamlit per Demo e Validazione
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Sistema Collocamento Mirato",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class JobMatchingDemo:
    def __init__(self):
        from geopy.geocoders import Nominatim
        self.geolocator = Nominatim(user_agent="job_matching_system")
        self.loc_cache = {}
        self.load_data()
        self.load_models()
    
    def geocode_with_cache(self, address):
        """Geocode with caching to avoid repeated API calls"""
        if address in self.loc_cache:
            return self.loc_cache[address]
        
        try:
            location = self.geolocator.geocode(address, timeout=10)
            import time
            time.sleep(0.5)  # Rate limiting
            coords = (location.latitude, location.longitude) if location else (np.nan, np.nan)
        except Exception as e:
            print(f"⚠️ Geocoding failed for {address}: {e}")
            coords = (np.nan, np.nan)
        
        self.loc_cache[address] = coords
        return coords
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on earth"""
        if any(pd.isnull([lat1, lon1, lat2, lon2])):
            return np.nan
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def load_data(self):
        """Load datasets from real files or generate demo data"""
        try:
            # Check for real files
            candidates_path = 'data/processed/Dataset_Candidati_Aggiornato_Extended.csv'
            companies_path = 'data/processed/Dataset_Aziende_con_Stima_Assunzioni_Extended.csv'
            
            if os.path.exists(candidates_path):
                print("✅ LOADING REAL CANDIDATE DATA")
                self.df_candidates = pd.read_csv(candidates_path)
                print(f"📊 Candidate columns: {list(self.df_candidates.columns)}")
                
                # Create ID_Candidato if missing
                if 'ID_Candidato' not in self.df_candidates.columns:
                    self.df_candidates['ID_Candidato'] = [f'CAND_{i:03d}' for i in range(1, len(self.df_candidates) + 1)]
                
                # Parse coordinates if they exist as strings
                if 'Coord_Residenza' in self.df_candidates.columns:
                    self.df_candidates = self._parse_coordinates(self.df_candidates, 'Coord_Residenza', 'Lat', 'Lon')
                    
            else:
                print("⚠️ GENERATING DEMO CANDIDATE DATA")
                self.df_candidates = self.create_demo_candidates()
            
            if os.path.exists(companies_path):
                print("✅ LOADING REAL COMPANY DATA")
                self.df_companies = pd.read_csv(companies_path)
                print(f"🏢 Company columns: {list(self.df_companies.columns)}")
                
                # Parse coordinates if they exist as strings  
                if 'Coord_Attività' in self.df_companies.columns:
                    self.df_companies = self._parse_coordinates(self.df_companies, 'Coord_Attività', 'Lat_a', 'Lon_a')
                elif 'Area di Attività' in self.df_companies.columns:
                    # Geocode company addresses if no coordinates
                    print("🗺️ GEOCODING COMPANY ADDRESSES...")
                    coords = self.df_companies['Area di Attività'].apply(self.geocode_with_cache)
                    self.df_companies[['Lat_a', 'Lon_a']] = pd.DataFrame(coords.tolist(), index=self.df_companies.index)
            else:
                print("⚠️ GENERATING DEMO COMPANY DATA")
                self.df_companies = self.create_demo_companies()
                
        except Exception as e:
            print(f"❌ Loading error: {e}")
            print("🔄 Switching to demo data")
            self.df_candidates = self.create_demo_candidates()
            self.df_companies = self.create_demo_companies()
    
    def _parse_coordinates(self, df, coord_col, lat_col, lon_col):
        """Parse coordinate strings like '(45.5232, 10.9473)' into separate lat/lon columns"""
        def parse_coord_string(coord_str):
            try:
                if pd.isna(coord_str) or coord_str == '':
                    return np.nan, np.nan
                # Remove parentheses and split
                clean_str = str(coord_str).strip('()')
                lat, lon = clean_str.split(',')
                return float(lat.strip()), float(lon.strip())
            except:
                return np.nan, np.nan
        
        coords = df[coord_col].apply(parse_coord_string)
        df[[lat_col, lon_col]] = pd.DataFrame(coords.tolist(), index=df.index)
        return df
    
    def create_demo_candidates(self):
        """Create demo candidate dataset with consistent array lengths"""
        np.random.seed(42)  # For reproducibility
        n_candidates = 20
        
        data = {
            'ID_Candidato': [f'CAND_{i:03d}' for i in range(1, n_candidates + 1)],
            'Area di Residenza': np.random.choice(['Verona', 'Sommacampagna', 'Villafranca di Verona', 'Negrar'], n_candidates),
            'Titolo di Studio': np.random.choice(['Diploma', 'Laurea', 'Master', 'Licenza Media'], n_candidates),
            'Tipo di Disabilità': np.random.choice(['Motoria', 'Sensoriale', 'Intellettiva', 'Psichica'], n_candidates),
            'Score Attitudine al Collocamento': np.random.uniform(0.3, 1.0, n_candidates),
            'Years_of_Experience': np.random.randint(0, 15, n_candidates),
            'Durata Disoccupazione': np.random.randint(0, 36, n_candidates),
            'Esclusioni': np.random.choice(['Turni notturni', 'Lavori in quota', 'Mansioni di responsabilità', ''], n_candidates)
        }
        return pd.DataFrame(data)
    
    def create_demo_companies(self):
        """Create demo company dataset with more variability"""
        np.random.seed(42)  # For reproducibility
        n_companies = 30
        
        settori = ['Servizi IT', 'Commercio al dettaglio', 'Produzione industriale', 
                  'Logistica e trasporti', 'Educazione', 'Sanità', 'Consulenza aziendale',
                  'Ristorazione', 'Agricoltura', 'Costruzioni']
        
        aree = ['Verona', 'Sommacampagna', 'Villafranca di Verona', 'Bussolengo', 
                'Legnago', 'San Bonifacio', 'Bardolino', 'Negrar']
        
        compatibilita_testi = [
            'Elevata compatibilità con limitazioni fisiche, ambiente accessibile',
            'Compatibilità moderata con limitazioni visive, supporto tecnologico',
            'Elevata compatibilità per lavoro da remoto, flessibilità orari',
            'Compatibilità con orari flessibili, supporto personalizzato',
            'Compatibilità moderata con limitazioni intellettive, tutoraggio',
            'Bassa compatibilità con mansioni di responsabilità',
            'Compatibilità con uso di mezzi semoventi',
            'Elevata compatibilità con limitazioni uditive',
            'Ambiente di lavoro protetto, supervisione costante',
            'Possibilità di lavoro part-time, ambiente tranquillo'
        ]
        
        data = {
            'Nome Azienda': [f'Azienda_{i:03d}' for i in range(1, n_companies + 1)],
            'Area di Attività': np.random.choice(aree, n_companies),
            'Tipo di Attività': np.random.choice(settori, n_companies),
            'Numero Dipendenti': np.random.randint(5, 800, n_companies),
            'Remote': np.random.choice([0, 1], n_companies, p=[0.6, 0.4]),
            'Certification': np.random.choice([0, 1], n_companies, p=[0.7, 0.3]),
            'Retention_Rate': np.random.beta(2, 2, n_companies),  # More realistic distribution
            'Posizioni Aperte': np.random.randint(1, 8, n_companies),
            'Compatibilità': np.random.choice(compatibilita_testi, n_companies)
        }
        
        df = pd.DataFrame(data)
        
        # Company size based on number of employees
        def map_size(n):
            if n < 50: return 'small'
            elif n < 250: return 'medium'
            else: return 'large'
        
        df['Company_Size'] = df['Numero Dipendenti'].apply(map_size)
        
        return df
    
    def load_models(self):
        """Load trained models if available"""
        self.models = {}
        model_dir = 'results'
        
        if os.path.exists(model_dir):
            for model_file in os.listdir(model_dir):
                if model_file.endswith('.joblib'):
                    try:
                        model_name = model_file.replace('.joblib', '')
                        self.models[model_name] = joblib.load(os.path.join(model_dir, model_file))
                    except:
                        pass
    
    def calculate_compatibility_score(self, exclusions, company_compatibility):
        """Calcola score di compatibilità"""
        if not exclusions or pd.isna(exclusions):
            return 1.0
        
        exclusions_list = [e.strip().lower() for e in str(exclusions).split(',')]
        company_text = str(company_compatibility).lower()
        
        # Semplice matching testuale
        conflicts = sum(1 for excl in exclusions_list if excl in company_text)
        return max(0.0, 1.0 - (conflicts * 0.3))
    
    def calculate_distance(self, candidate_area, company_area, candidate_coords=None, company_coords=None):
        """Calculate real distance using coordinates from dataset"""
        
        # If we have real coordinates, use them
        if (candidate_coords is not None and company_coords is not None and 
            not any(pd.isna([candidate_coords[0], candidate_coords[1], company_coords[0], company_coords[1]]))):
            
            return self.haversine(candidate_coords[0], candidate_coords[1], 
                                company_coords[0], company_coords[1])
        
        # Fallback to area-based distance calculation
        base_distances = {
            ('Verona', 'Verona'): 0, ('Verona', 'Sommacampagna'): 8, ('Verona', 'Villafranca di Verona'): 15,
            ('Verona', 'Bussolengo'): 18, ('Verona', 'Legnago'): 30, ('Verona', 'San Bonifacio'): 25,
            ('Verona', 'Bardolino'): 28, ('Verona', 'Negrar'): 12,
            ('Sommacampagna', 'Verona'): 8, ('Sommacampagna', 'Sommacampagna'): 0,
            ('Sommacampagna', 'Villafranca di Verona'): 12, ('Sommacampagna', 'Bussolengo'): 15,
            ('Sommacampagna', 'Legnago'): 35, ('Sommacampagna', 'San Bonifacio'): 28,
            ('Sommacampagna', 'Bardolino'): 22, ('Sommacampagna', 'Negrar'): 18,
            ('Villafranca di Verona', 'Verona'): 15, ('Villafranca di Verona', 'Sommacampagna'): 12,
            ('Villafranca di Verona', 'Villafranca di Verona'): 0, ('Villafranca di Verona', 'Bussolengo'): 20,
            ('Villafranca di Verona', 'Legnago'): 25, ('Villafranca di Verona', 'San Bonifacio'): 18,
            ('Villafranca di Verona', 'Bardolino'): 35, ('Villafranca di Verona', 'Negrar'): 22,
            ('Negrar', 'Verona'): 12, ('Negrar', 'Legnago'): 35, ('Negrar', 'Sommacampagna'): 18,
            ('Negrar', 'Villafranca di Verona'): 22, ('Negrar', 'Negrar'): 0,
            ('Legnago', 'Verona'): 30, ('Legnago', 'Legnago'): 0, ('Legnago', 'Negrar'): 35,
            ('Bussolengo', 'Verona'): 18, ('Bussolengo', 'Bussolengo'): 0,
        }
        
        # Get base distance
        base_dist = base_distances.get((candidate_area, company_area), 25)
        
        # Add small variation for realism
        variation = 1 + (np.random.random() - 0.5) * 0.2  # ±10%
        return max(1, round(base_dist * variation, 1))
    
    def find_matches(self, candidate_data, top_k=5, distance_threshold=30, attitude_threshold=0.3):
        """Find matching companies for candidate using REAL data"""
        matches = []
        
        # Check global filters
        if candidate_data['Score Attitudine al Collocamento'] < attitude_threshold:
            st.warning(f"⚠️ Candidate attitude {candidate_data['Score Attitudine al Collocamento']:.2f} below threshold {attitude_threshold}")
            return []
        
        # Get candidate coordinates if available
        candidate_coords = None
        if hasattr(self, 'df_candidates'):
            # Find candidate in dataframe to get coordinates
            candidate_mask = (
                (self.df_candidates['Area di Residenza'] == candidate_data['Area di Residenza']) &
                (self.df_candidates['Tipo di Disabilità'] == candidate_data['Tipo di Disabilità']) &
                (abs(self.df_candidates['Score Attitudine al Collocamento'] - candidate_data['Score Attitudine al Collocamento']) < 0.01)
            )
            if candidate_mask.any():
                candidate_row = self.df_candidates[candidate_mask].iloc[0]
                if 'Lat' in candidate_row and 'Lon' in candidate_row:
                    candidate_coords = (candidate_row['Lat'], candidate_row['Lon'])
        
        print(f"🔍 DEBUG: Candidate coordinates: {candidate_coords}")
        print(f"🔍 DEBUG: Distance threshold: {distance_threshold} km")
        
        companies_within_range = 0
        
        for _, company in self.df_companies.iterrows():
            # Get company coordinates
            company_coords = None
            if 'Lat_a' in company and 'Lon_a' in company:
                company_coords = (company['Lat_a'], company['Lon_a'])
            
            # Calculate REAL compatibility using actual data
            compatibility = self.calculate_compatibility_score(
                candidate_data['Esclusioni'], 
                company.get('Compatibilità', '')
            )
            
            # Calculate REAL distance using coordinates
            distance = self.calculate_distance(
                candidate_data['Area di Residenza'],
                company['Area di Attività'],
                candidate_coords,
                company_coords
            )
            
            print(f"🔍 DEBUG: {company['Nome Azienda']} - Distance: {distance:.1f} km")
            
            # DISTANCE FILTER
            if distance > distance_threshold:
                continue  # Skip companies too far away
            
            companies_within_range += 1
            
            attitude_score = candidate_data['Score Attitudine al Collocamento']
            
            # Additional factors for variability
            distance_factor = max(0, 1 - distance/40)
            experience_bonus = min(0.2, candidate_data['Years_of_Experience'] / 20)
            unemployment_penalty = max(0, 1 - candidate_data['Durata Disoccupazione'] / 60)
            
            # Company bonuses using REAL data
            remote_bonus = 0.05 if company.get('Remote', 0) == 1 else 0
            cert_bonus = 0.03 if company.get('Certification', 0) == 1 else 0
            
            # Use REAL retention rate from data
            retention_rate = company.get('Retention_Rate', 0.5)
            if pd.isna(retention_rate):
                retention_rate = 0.5
            
            # Size bonus using REAL company size
            size_bonus = {
                'small': 0.02, 
                'medium': 0.05, 
                'large': 0.03
            }.get(company.get('Company_Size', 'medium'), 0.03)
            
            # Final score with more components
            final_score = (
                0.35 * compatibility +
                0.25 * distance_factor +
                0.20 * attitude_score +
                0.10 * retention_rate +
                0.05 * experience_bonus +
                0.03 * unemployment_penalty +
                0.015 * remote_bonus +
                0.015 * cert_bonus +
                0.01 * size_bonus
            )
            
            # Add small random variation (±2%)
            random_factor = 1 + (np.random.random() - 0.5) * 0.04
            final_score *= random_factor
            
            matches.append({
                'Nome Azienda': company['Nome Azienda'],
                'Tipo di Attività': company['Tipo di Attività'],
                'Score Finale': round(final_score * 100, 1),
                'Compatibilità': round(compatibility * 100, 1),
                'Distanza (km)': round(distance, 1),  # Make sure distance is stored correctly
                'Dipendenti': company['Numero Dipendenti'],
                'Remote': 'Sì' if company.get('Remote', 0) == 1 else 'No',
                'Certificazione': 'Sì' if company.get('Certification', 0) == 1 else 'No',
                'Posizioni Aperte': company.get('Posizioni Aperte', 1),
                'Area': company['Area di Attività'],
                'Retention_Rate': round(retention_rate * 100, 1),
                'Debug_Distance_Factor': round(distance_factor, 3),
                'Debug_Experience_Bonus': round(experience_bonus, 3),
                'Debug_Raw_Distance': distance,  # Store raw distance for debugging
                'Debug_Raw_Compatibility': compatibility  # Store raw compatibility for debugging
            })
        
        print(f"🔍 DEBUG: Found {companies_within_range} companies within {distance_threshold} km")
        
        if matches:
            # Debug: Show ALL matches first before sorting
            print("🔍 DEBUG: All matches before sorting:")
            for i, match in enumerate(matches[:10]):  # Show first 10
                print(f"  {match['Nome Azienda']}: Score={match['Score Finale']:.1f}%, Distance={match['Debug_Raw_Distance']:.1f}km, Compat={match['Debug_Raw_Compatibility']:.3f}")
            
            # Debug: Show top 5 after sorting
            sorted_matches = sorted(matches, key=lambda x: x['Score Finale'], reverse=True)
            print("🔍 DEBUG: Top 5 after sorting:")
            for i, match in enumerate(sorted_matches[:5]):
                print(f"  {match['Nome Azienda']}: Score={match['Score Finale']:.1f}%, Distance={match['Debug_Raw_Distance']:.1f}km, Compat={match['Debug_Raw_Compatibility']:.3f}")
        
        if not matches:
            st.warning(f"❌ No companies found within {distance_threshold} km with minimum attitude {attitude_threshold}")
            return []
        
        return sorted(matches, key=lambda x: x['Score Finale'], reverse=True)[:top_k]

def main():
    demo = JobMatchingDemo()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Sistema di Raccomandazione per Collocamento Mirato</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configurazione Sistema")
    
    # Model selection
    if demo.models:
        selected_model = st.sidebar.selectbox(
            "Modello ML", 
            options=list(demo.models.keys()),
            index=0
        )
        st.sidebar.success(f"✅ Modello {selected_model} caricato")
    else:
        st.sidebar.warning("⚠️ Usando sistema a regole (nessun modello ML trovato)")
    
    # Thresholds
    st.sidebar.subheader("⚙️ Soglie Sistema")
    attitude_threshold = st.sidebar.slider("Soglia Attitudine", 0.0, 1.0, 0.3, 0.1)
    distance_threshold = st.sidebar.slider("Distanza Max (km)", 5, 50, 30, 5)
    top_k = st.sidebar.slider("Top Raccomandazioni", 3, 10, 5, 1)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Ricerca Candidato", "📊 Analytics", "📋 Dataset", "ℹ️ Info Sistema"])
    
    with tab1:
        st.header("🔍 Trova Aziende per Candidato")
        
        # Проверяем доступные колонки для селекторов
        areas = demo.df_candidates['Area di Residenza'].unique() if 'Area di Residenza' in demo.df_candidates.columns else ['Verona', 'Sommacampagna', 'Villafranca di Verona']
        education = demo.df_candidates['Titolo di Studio'].unique() if 'Titolo di Studio' in demo.df_candidates.columns else ['Licenza Media', 'Diploma', 'Laurea', 'Master']
        disabilities = demo.df_candidates['Tipo di Disabilità'].unique() if 'Tipo di Disabilità' in demo.df_candidates.columns else ['Motoria', 'Sensoriale', 'Intellettiva', 'Psichica']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📝 Dati Candidato")
            
            # Candidate selection
            if st.checkbox("Usa candidato esistente"):
                # Проверяем, есть ли ID_Candidato, если нет - используем индекс
                if 'ID_Candidato' in demo.df_candidates.columns:
                    candidate_ids = demo.df_candidates['ID_Candidato'].tolist()
                    selected_id = st.selectbox("Seleziona Candidato", candidate_ids)
                    candidate_row = demo.df_candidates[demo.df_candidates['ID_Candidato'] == selected_id].iloc[0]
                else:
                    # Если нет ID, используем индексы
                    candidate_indices = demo.df_candidates.index.tolist()
                    selected_idx = st.selectbox("Seleziona Candidato", 
                                               [f"Candidato {i+1}" for i in candidate_indices])
                    idx = int(selected_idx.split()[1]) - 1
                    candidate_row = demo.df_candidates.iloc[idx]
                
                # Адаптивное извлечение данных
                candidate_data = {}
                
                # DEBUG: Print original value
                original_score = candidate_row.get('Score Attitudine al Collocamento', 0.7)
                selected_identifier = selected_idx if 'selected_idx' in locals() else selected_id
                print(f"🔍 DEBUG: Candidate {selected_identifier}")
                print(f"🔍 DEBUG: Original score in CSV: {original_score}")
                
                # Required fields with fallback values
                candidate_data['Area di Residenza'] = candidate_row.get('Area di Residenza', 'Verona')
                candidate_data['Titolo di Studio'] = candidate_row.get('Titolo di Studio', 'Diploma')
                candidate_data['Tipo di Disabilità'] = candidate_row.get('Tipo di Disabilità', 'Motoria')
                candidate_data['Score Attitudine al Collocamento'] = original_score  # Use original value
                candidate_data['Years_of_Experience'] = candidate_row.get('Years_of_Experience', 5)
                candidate_data['Durata Disoccupazione'] = candidate_row.get('Durata Disoccupazione', 6)
                candidate_data['Esclusioni'] = candidate_row.get('Esclusioni', '')
                
                print(f"🔍 DEBUG: Final score used: {candidate_data['Score Attitudine al Collocamento']}")
                
            else:
                # Manual input with adaptive options
                candidate_data = {
                    'Area di Residenza': st.selectbox("Area Residenza", areas),
                    'Titolo di Studio': st.selectbox("Titolo Studio", education),
                    'Tipo di Disabilità': st.selectbox("Tipo Disabilità", disabilities),
                    'Score Attitudine al Collocamento': st.slider("Attitudine", 0.0, 1.0, 0.7, 0.1),
                    'Years_of_Experience': st.number_input("Anni Esperienza", 0, 20, 5),
                    'Durata Disoccupazione': st.number_input("Mesi Disoccupazione", 0, 60, 6),
                    'Esclusioni': st.text_input("Esclusioni (separate da virgola)", 
                        "Turni notturni, Lavori in quota")
                }
            
            # Show candidate summary
            st.markdown("### 👤 Profilo Candidato")
            for key, value in candidate_data.items():
                if key == 'Score Attitudine al Collocamento':
                    st.metric(key, f"{value:.2f}")
                else:
                    st.write(f"**{key}**: {value}")
        
        with col2:
            st.subheader("🏢 Raccomandazioni Aziende")
            
            if st.button("🔄 Trova Aziende Compatibili", type="primary"):
                # Получаем текущие настройки из sidebar
                current_attitude_threshold = attitude_threshold
                current_distance_threshold = distance_threshold
                
                matches = demo.find_matches(
                    candidate_data, 
                    top_k=top_k,
                    distance_threshold=current_distance_threshold,
                    attitude_threshold=current_attitude_threshold
                )
                
                if matches:
                    st.success(f"✅ Found {len(matches)} compatible companies within {current_distance_threshold} km")
                    
                    # Display matches
                    for i, match in enumerate(matches, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{i}. {match['Nome Azienda']} 
                                    <span style="color: #1f77b4;">({match['Score Finale']:.1f}%)</span>
                                </h4>
                                <div style="display: flex; gap: 20px; margin-top: 10px;">
                                    <div><strong>Sector:</strong> {match['Tipo di Attività']}</div>
                                    <div><strong>Distance:</strong> {match['Distanza (km)']} km</div>
                                    <div><strong>Employees:</strong> {match['Dipendenti']}</div>
                                </div>
                                <div style="display: flex; gap: 20px; margin-top: 5px;">
                                    <div><strong>Compatibility:</strong> {match['Compatibilità']:.1f}%</div>
                                    <div><strong>Remote:</strong> {match['Remote']}</div>
                                    <div><strong>Positions:</strong> {match['Posizioni Aperte']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    # Visualization
                    if matches:  # Только если есть результаты
                        st.subheader("📈 Analisi Visuale")
                        
                        # Score distribution
                        scores_df = pd.DataFrame(matches)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(
                                scores_df, 
                                x='Nome Azienda', 
                                y='Score Finale',
                                title="Score Finale per Azienda",
                                color='Score Finale',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.scatter(
                                scores_df,
                                x='Distanza (km)',
                                y='Compatibilità',
                                size='Score Finale',
                                hover_name='Nome Azienda',
                                title="Compatibilità vs Distanza"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("❌ No companies found with current filters. Try increasing maximum distance or reducing attitude threshold.")
    
    with tab2:
        st.header("📊 Analytics del Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Candidati Totali", len(demo.df_candidates))
        with col2:
            st.metric("🏢 Aziende Totali", len(demo.df_companies))
        with col3:
            avg_attitude = demo.df_candidates['Score Attitudine al Collocamento'].mean()
            st.metric("📈 Attitudine Media", f"{avg_attitude:.2f}")
        with col4:
            total_positions = demo.df_companies['Posizioni Aperte'].sum()
            st.metric("💼 Posizioni Aperte", total_positions)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                demo.df_candidates, 
                x='Tipo di Disabilità',
                title="Distribuzione Tipi di Disabilità"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                demo.df_companies, 
                x='Tipo di Attività',
                title="Distribuzione Settori Aziende"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📋 Visualizzazione Dataset")
        
        dataset_choice = st.radio("Seleziona Dataset", ["Candidati", "Aziende"])
        
        if dataset_choice == "Candidati":
            st.subheader("👥 Dataset Candidati")
            st.dataframe(demo.df_candidates, use_container_width=True)
            
            # Download
            csv = demo.df_candidates.to_csv(index=False)
            st.download_button(
                label="📥 Scarica CSV Candidati",
                data=csv,
                file_name="candidati.csv",
                mime="text/csv"
            )
        
        else:
            st.subheader("🏢 Dataset Aziende")
            st.dataframe(demo.df_companies, use_container_width=True)
            
            # Download
            csv = demo.df_companies.to_csv(index=False)
            st.download_button(
                label="📥 Scarica CSV Aziende",
                data=csv,
                file_name="aziende.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.header("ℹ️ Informazioni Sistema")
        
        st.markdown("""
        ### 🎯 Sistema di Raccomandazione per Collocamento Mirato
        
        Questo sistema utilizza tecniche di **Machine Learning** e **algoritmi di matching** 
        per supportare il processo di collocamento mirato delle persone con disabilità.
        
        #### 🔧 Caratteristiche Principali:
        - **Matching Intelligente**: Analisi automatica della compatibilità candidato-azienda
        - **Filtri Geografici**: Calcolo distanze e preferenze di mobilità
        - **Score Multidimensionale**: Combinazione di fattori di attitudine, compatibilità e logistici
        - **Interfaccia Intuitiva**: Dashboard per operatori CPI/SIL
        
        #### 📊 Algoritmo di Scoring:
        ```
        Score Finale = 0.4 × Compatibilità + 0.3 × Distanza + 0.2 × Attitudine + 0.1 × Retention Rate
        ```
        
        #### 🚀 Stato Sviluppo:
        - ✅ **Prototipo Funzionante**: Sistema completo con demo data
        - ✅ **Validazione Esperta**: Approvato da Centro per l'Impiego di Villafranca
        - 🔄 **In Attesa**: Integrazione dati reali da CPI/SIL
        - 📈 **Prossimi Passi**: Deploy in produzione e A/B testing
        
        #### 👨‍💻 Sviluppato da:
        **Michele Melchiori** - Tesi di Laurea in Ingegneria Informatica  
        **Relatore**: Prof. Oleksandr Kuznetsov  
        **Partner**: Centro per l'Impiego di Villafranca di Verona
        """)
        
        # System status
        st.subheader("🔍 Stato Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ Interface: Operativo")
            st.success("✅ Dataset Demo: Caricato")
            st.info("ℹ️ Algoritmi ML: Demo Mode")
        
        with col2:
            if demo.models:
                st.success(f"✅ Modelli ML: {len(demo.models)} caricati")
            else:
                st.warning("⚠️ Modelli ML: Non trovati (usando regole)")
            
            st.info(f"📅 Ultimo Aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

if __name__ == "__main__":
    main()