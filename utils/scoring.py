# utils/scoring.py
# -*- coding: utf-8 -*-
"""
Enhanced scoring logic for candidate-company matching.
Includes:
- Geocoding with caching
- Haversine distance
- Compatibility scoring
- Final training dataset generator
"""

import time
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedScoringSystem:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="job_matcher_it")
        self.loc_cache = {}
        self.thresholds = {
            'attitude_min': 0.3,
            'compatibility_min': 0.5,
            'distance_max': 40.0
        }

    def geocode_with_cache(self, address):
        if address in self.loc_cache:
            return self.loc_cache[address]

        try:
            location = self.geolocator.geocode(address, timeout=10)
            time.sleep(0.5)
            coords = (location.latitude, location.longitude) if location else (np.nan, np.nan)
        except:
            coords = (np.nan, np.nan)

        self.loc_cache[address] = coords
        return coords

    def haversine(self, lat1, lon1, lat2, lon2):
        if any(pd.isnull([lat1, lon1, lat2, lon2])):
            return np.nan
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * 6371.0 * np.arcsin(np.sqrt(a))

    def compatibility_score(self, exclusions, company_text):
        if pd.isna(exclusions) or not company_text:
            return 1.0

        exclusion_list = [e.strip().lower() for e in str(exclusions).split(',') if e.strip()]
        all_texts = exclusion_list + [company_text.lower()]

        try:
            italian_stop_words = [
                'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'il', 'lo', 'la', 'i',
                'gli', 'le', 'un', 'una', 'uno', 'e', 'o', 'ma', 'se', 'che', 'chi', 'cui'
            ]
            vectorizer = TfidfVectorizer(
                stop_words=italian_stop_words,
                token_pattern=r'\b[a-zA-Zàèéìòù]+\b',
                lowercase=True
            )
            tfidf = vectorizer.fit_transform(all_texts)
            company_vec = tfidf[-1]
            sims = [cosine_similarity(tfidf[i], company_vec)[0][0] for i in range(len(exclusion_list))]
            score = 1.0 - (0.7 * max(sims) + 0.3 * np.mean(sims))
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def generate_enhanced_training_data(self, df_cand, df_az):
        print("🔄 Generating combinations...")

        if 'Lat' not in df_cand:
            coords = df_cand['Area di Residenza'].apply(self.geocode_with_cache)
            df_cand[['Lat', 'Lon']] = pd.DataFrame(coords.tolist(), index=df_cand.index)

        if 'Lat_a' not in df_az:
            coords = df_az['Area di Attività'].apply(self.geocode_with_cache)
            df_az[['Lat_a', 'Lon_a']] = pd.DataFrame(coords.tolist(), index=df_az.index)

        records = []
        for _, c in df_cand.iterrows():
            for _, a in df_az.iterrows():
                try:
                    attitude = c['Score Attitudine al Collocamento']
                    compat = self.compatibility_score(c['Esclusioni'], a['Compatibilità'])
                    dist = self.haversine(c['Lat'], c['Lon'], a['Lat_a'], a['Lon_a'])

                    attitude_factor = min(1.0, attitude / self.thresholds['attitude_min'])
                    compat_factor = min(1.0, compat / self.thresholds['compatibility_min'])
                    distance_factor = max(0.0, 1.0 - (dist / self.thresholds['distance_max'])) if not pd.isna(dist) else 0.5

                    remote_bonus = 0.1 if a['Remote'] == 1 else 0.0
                    cert_bonus = 0.1 if a['Certification'] == 1 else 0.0

                    prob = (
                        0.3 * attitude_factor + 0.4 * compat_factor + 0.2 * distance_factor +
                        0.05 * a['Retention_Rate'] + 0.025 * remote_bonus + 0.025 * cert_bonus
                    )

                    outcome = 1 if (prob > 0.6 and np.random.random() < prob) else 0

                    record = {
                        'outcome': outcome,
                        'attitude_score': attitude,
                        'years_experience': c['Years_of_Experience'],
                        'unemployment_duration': c['Durata Disoccupazione'],
                        'compatibility_score': compat,
                        'distance_km': dist if not pd.isna(dist) else self.thresholds['distance_max'],
                        'company_size': a['Numero Dipendenti'],
                        'retention_rate': a['Retention_Rate'],
                        'remote_work': a['Remote'],
                        'certification': a['Certification'],
                        'match_probability': prob
                    }

                    # Encodings (education, disability, sector)
                    for e in df_cand['Titolo di Studio'].unique():
                        record[f'edu_{e.lower().replace(" ", "_")}'] = int(c['Titolo di Studio'] == e)
                    for d in df_cand['Tipo di Disabilità'].unique():
                        record[f'dis_{d.lower().replace(" ", "_")}'] = int(c['Tipo di Disabilità'] == d)
                    for s in df_az['Tipo di Attività'].unique():
                        record[f'sector_{s.lower().replace(" ", "_")}'] = int(a['Tipo di Attività'] == s)

                    records.append(record)
                except Exception as ex:
                    print(f"⚠️ Error on pair: {ex}")
                    continue

        print(f"✅ Generated {len(records)} candidate-company pairs")
        return pd.DataFrame(records)
