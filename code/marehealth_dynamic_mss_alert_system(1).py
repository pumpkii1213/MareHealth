

import os
import sys
import json
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ==================== configuration parameter ====================

# Time mapping: raw code → actual date string
TIME_MAP = {
    1015: '2025-10-15',
    1020: '2025-10-20',
    1027: '2025-10-27',
    1102: '2025-11-02',
    1108: '2025-11-08'
}

# Time mapping: raw code → relative days (for slope calculation)
DAY_MAP = {
    1015: 0,
    1020: 5,
    1027: 12,
    1102: 18,
    1108: 24
}

# ---------------- MSS-physo ----------------

# MSS weights (obtained from LASSO regression)
MSS_WEIGHTS = {
    68.0575:  -0.967339,   # Negative
    84.0378:   0.561247,   # Positive
    94.0448:   0.243156,   # Positive
    73.0647:  -0.206175,   # Negative
    120.102:   0.186569,   # Positive
    55.044:   -0.034769,   # Negative
    62.0302:  -0.029404,   # Negative
}

# MSS key metabolites (m/z)
CORE_MZ = [68.0575, 84.0378, 94.0448, 73.0647, 120.102, 55.044, 62.0302]

# Clinical thresholds (for symptom risk assessment)
CLINICAL_THRESHOLDS = {
    'fidget':                    {'threshold': 2, 'auc': 0.867, 'symptom': 'Fidget'},
    'respiratory_symptom':       {'threshold': 1, 'auc': 0.837, 'symptom': 'Respiratory Symptom'},
    'dizziness_palpitations':    {'threshold': 2, 'auc': 0.831, 'symptom': 'Dizziness/Palpitations'},
    'physical_symptom':          {'threshold': 3, 'auc': 0.777, 'symptom': 'Physical Symptom'},
    'digestive_problem':         {'threshold': 2, 'auc': 0.748, 'symptom': 'Digestive Problem'},
    'insomnia':                  {'threshold': 2, 'auc': 0.706, 'symptom': 'Insomnia'},
}

# Somatic symptom column name mapping
SYMPTOM_COLS = {
    'physical_symptom': 'physical symptom',
    'insomnia': 'Insomnia or excessive sleepiness',
    'dizziness_palpitations': 'Dizziness, palpitations or difficulty in breathing',
    'fidget': 'fidget',
    'digestive_problem': 'digestive problem',
    'respiratory_symptom': 'respiratory symptom',
}

# ---------------- MSS-psycho ----------------

# Psychological state model: weights are raw ElasticNet coefficients (questionnaire score/abundance units)
# Δpsy = Σ w·Δmz，Δmz = current abund − previous
PSYCHO_MODELS = {
    'fatigue': {
        'label': 'Fatigue',
        'questionnaire_col': 'fatigue_total',
        'threshold': 22, 'cmp': '>=',
        'threshold_desc': 'fatigue >= 22 (clinically significant fatigue)',
        'weights': {
            55.044:  -48.4787,   # N-methylene-ethenamine
            94.0448:  10.5418,   # Phenol
            73.0647:  -1.6659,   # 1-Butanol
            59.0754:   0.5797,   # Trimethylamine
            68.0575:  -0.1340,   # Isoprene
        },
        'resid_sd': 6.619,
        'auc': 0.827, 'loso_auc': 0.810, 'incr_auc_loso': 0.031,
        'confidence': 'HIGH',
    },
    'emotion': {
        'label': 'Emotion',
        'questionnaire_col': 'emotion_total',
        'threshold': 6, 'cmp': '>=',
        'threshold_desc': 'emotion >= 6 (depression/anxiety risk)',
        'weights': {
            84.0378:  32.9770,   # 2(5H)-Furanone
            120.102:  -21.5456,  # Cumene
            79.0268:  -20.4367,  # Pyridine
            94.0448:   14.5030,  # Phenol
            62.0302:   13.8064,  # Dimethylsulfide
            134.108:   11.8629,  # p-Cymene
            76.0339:    8.5124,  # Glycine
            73.0647:    5.2091,  # 1-Butanol
            55.044:    -3.9743,  # N-methylene-ethenamine
            59.0754:    2.0079,  # Trimethylamine
            68.0575:    0.2149,  # Isoprene
        },
        'resid_sd': 2.311,
        'auc': 0.827, 'loso_auc': 0.784, 'incr_auc_loso': 0.028,
        'confidence': 'MEDIUM',
    },
    'stress': {
        'label': 'Stress',
        'questionnaire_col': 'stress_total',
        'threshold': 20, 'cmp': '>',
        'threshold_desc': 'stress > 20 (high stress / stress response risk)',
        'weights': {
            55.044:  -9.6611,    # N-methylene-ethenamine
            94.0448:  8.0065,    # Phenol
            68.0575: -1.7774,    # Isoprene
            59.0754:  1.3201,    # Trimethylamine
        },
        'resid_sd': 4.491,
        'auc': 0.851, 'loso_auc': 0.831, 'incr_auc_loso': -0.001,
        'confidence': 'LOW',
    },
}

# Psychological-state questionnaire population statistics (fallback baseline when no questionnaire history exists)
PSYCHO_POP_STATS = {
    'fatigue': {'mean': 16.90, 'std': 7.99},
    'emotion': {'mean': 2.88,  'std': 2.74},
    'stress':  {'mean': 9.14,  'std': 6.25},
}

# All m/z values involved in MSS-psycho (used for extraction from input rows)
PSYCHO_MZ = sorted({mz for m in PSYCHO_MODELS.values() for mz in m['weights']})

# Cluster centroid profile (mean_raw_slope, based on 20 metabolites)
CLUSTER_PROFILES = {
    1: {  # acute stress–sensitive subtype
        'description': 'High-Stress Responder',
        'risk_bias': 0.15,
        'phenol': 0.076468, '2(5H)-furanone': -0.013293,
        'N-methylene-ethenamine': -0.025883, 'cumene': -0.011943,
        'pyridine': 0.039938, 'N,N-dimethylformamide': 0.015804,
        'trimethylamine': 0.029573, 'p-Cymene': -0.004010,
        'pentanoic acid': -0.004843, '2-propenal': -0.002792,
        'isoprene': 0.010493, '1-butanol': 0.006903,
        'glycine': 0.024326, '2,6-dimethylpyridine': -0.001228,
        'dimethylsulfide': -0.001399, '6-Heptenoic acid': -0.001217,
        'cyclooctane': 0.002596, '2-ethylhexanal': -0.001471,
        'adamantyl alcohol': -0.001309, 'iso-valeraldehyde': -0.004262,
    },
    2: {  # high-resilience adaptive subtype
        'description': 'Low-Fluctuation Stable',
        'risk_bias': -0.10,
        'phenol': 0.064742, '2(5H)-furanone': -0.014125,
        'N-methylene-ethenamine': -0.026828, 'cumene': -0.014911,
        'pyridine': 0.019245, 'N,N-dimethylformamide': 0.014454,
        'trimethylamine': -0.014114, 'p-Cymene': -0.005468,
        'pentanoic acid': -0.005441, '2-propenal': -0.003238,
        'isoprene': 0.008625, '1-butanol': 0.005969,
        'glycine': 0.010155, '2,6-dimethylpyridine': -0.001930,
        'dimethylsulfide': -0.002604, '6-Heptenoic acid': -0.001411,
        'cyclooctane': 0.002269, '2-ethylhexanal': -0.001759,
        'adamantyl alcohol': -0.001588, 'iso-valeraldehyde': -0.004446,
    },
    3: {  # differentiated adaptive subtype
        'description': 'Mixed-Fluctuation',
        'risk_bias': 0.05,
        'phenol': 0.066998, '2(5H)-furanone': -0.011706,
        'N-methylene-ethenamine': -0.025820, 'cumene': -0.013888,
        'pyridine': 0.023111, 'N,N-dimethylformamide': 0.013409,
        'trimethylamine': -0.003490, 'p-Cymene': -0.005084,
        'pentanoic acid': -0.004154, '2-propenal': -0.002149,
        'isoprene': 0.007987, '1-butanol': 0.005009,
        'glycine': 0.012284, '2,6-dimethylpyridine': -0.001628,
        'dimethylsulfide': -0.002438, '6-Heptenoic acid': -0.001111,
        'cyclooctane': 0.002310, '2-ethylhexanal': -0.001295,
        'adamantyl alcohol': -0.001426, 'iso-valeraldehyde': -0.004220,
    }
}

# metabolite names
METABOLITE_NAMES = {
    94.0448: 'Phenol',
    84.0378: '2(5H)-Furanone',
    55.044: 'N-methylene-ethenamine',
    120.102: 'Cumene',
    79.0268: 'Pyridine',
    73.0534: 'DMF',
    59.0754: 'Trimethylamine',
    134.108: 'p-Cymene',
    102.077: 'Pentanoic acid',
    56.0136: '2-Propenal (acrolein)',
    68.0575: 'Isoprene',
    73.0647: '1-Butanol',
    76.0339: 'Glycine',
    107.101: '2,6-Dimethylpyridine',
    62.0302: 'Dimethylsulfide',
    128.073: '6-Heptenoic acid',
    112.105: 'Cyclooctane',
    128.103: '2-Ethylhexanal',
    152.194: 'Adamantyl alcohol',
    86.0828: 'iso-Valeraldehyde',
}

# Accurate population-level statistical parameters
POPULATION_STATS = {
    94.0448: {'mean': 0.145323, 'std': 0.129598, 'median': 0.117957, 'q25': 0.030384, 'q75': 0.226893},
    84.0378: {'mean': 0.022581, 'std': 0.022664, 'median': 0.006892, 'q25': 0.004229, 'q75': 0.045212},
    55.044:  {'mean': 0.058656, 'std': 0.049480, 'median': 0.038771, 'q25': 0.026898, 'q75': 0.056472},
    120.102: {'mean': 0.090129, 'std': 0.043964, 'median': 0.081323, 'q25': 0.057059, 'q75': 0.128440},
    79.0268: {'mean': 0.121614, 'std': 0.064365, 'median': 0.103765, 'q25': 0.075995, 'q75': 0.147879},
    73.0534: {'mean': 0.080375, 'std': 0.033240, 'median': 0.087493, 'q25': 0.063418, 'q75': 0.102934},
    59.0754: {'mean': 0.154949, 'std': 0.187950, 'median': 0.105576, 'q25': 0.067204, 'q75': 0.180208},
    134.108: {'mean': 0.046923, 'std': 0.015090, 'median': 0.044871, 'q25': 0.037078, 'q75': 0.057731},
    102.077: {'mean': 0.007961, 'std': 0.007971, 'median': 0.004311, 'q25': 0.002771, 'q75': 0.008560},
    56.0136: {'mean': 0.007033, 'std': 0.004512, 'median': 0.005242, 'q25': 0.003474, 'q75': 0.010326},
    68.0575: {'mean': 0.575951, 'std': 0.330805, 'median': 0.485128, 'q25': 0.339902, 'q75': 0.786728},
    73.0647: {'mean': 0.083532, 'std': 0.036250, 'median': 0.089294, 'q25': 0.067686, 'q75': 0.110489},
    76.0339: {'mean': 0.079802, 'std': 0.044175, 'median': 0.067033, 'q25': 0.053216, 'q75': 0.092805},
    107.101: {'mean': 0.025607, 'std': 0.013625, 'median': 0.021247, 'q25': 0.013753, 'q75': 0.036660},
    62.0302: {'mean': 0.028954, 'std': 0.019370, 'median': 0.025489, 'q25': 0.015759, 'q75': 0.038173},
    128.073: {'mean': 0.004163, 'std': 0.002013, 'median': 0.003403, 'q25': 0.002399, 'q75': 0.005735},
    112.105: {'mean': 0.014612, 'std': 0.005704, 'median': 0.013858, 'q25': 0.009920, 'q75': 0.018988},
    128.103: {'mean': 0.004322, 'std': 0.002449, 'median': 0.003522, 'q25': 0.002521, 'q75': 0.005124},
    152.194: {'mean': 0.011098, 'std': 0.003991, 'median': 0.010808, 'q25': 0.007731, 'q75': 0.014169},
    86.0828: {'mean': 0.013242, 'std': 0.009568, 'median': 0.009414, 'q25': 0.007619, 'q75': 0.014250},
}


# ==================== Core alert system ====================

class MareHealthMSSAlertSystem:
    """
    MareHealth v3.1
Supports two modes:

Single-point mode: For a subject’s first measurement, MSS-physo is assessed based on population statistics and cluster priors. MSS-psycho is unavailable because there is no previous measurement and Δmz cannot be computed.

Dynamic mode: For subjects with historical data, MSS-physo is assessed based on deviation from the individual baseline and trend. MSS-psycho predicts the psychological state as “previous questionnaire score + Σw·Δmz”.
    """

    def __init__(self, population_stats=None):
        self.subject_history = {}   # {subject_id: [measurement_records]}
        self.subject_clusters = {}  # {subject_id: cluster_id}
        self.population_stats = population_stats or POPULATION_STATS

        print(f"[INIT] MareHealth-dynamic MSS Alert System v3.1 initialized (MSS-physo + MSS-psycho)")
        print(f"[INIT] Population stats loaded for {len(self.population_stats)} metabolites")

    def set_subject_cluster(self, subject_id, cluster_id):
        """Manually set the subject's metabolic response subtype."""
        self.subject_clusters[int(subject_id)] = int(cluster_id)

    def parse_timestamp(self, time_val):
        """Parse a timestamp into an actual date -> (date_str, day_offset)."""
        if isinstance(time_val, float):
            time_val = int(time_val)

        if time_val in TIME_MAP:
            return TIME_MAP[time_val], DAY_MAP[time_val]

        time_str = str(int(time_val))
        if len(time_str) == 4:
            month = int(time_str[:2])
            day = int(time_str[2:])
            year = 2025 if month >= 10 else 2026
            date_str = f"{year}-{month:02d}-{day:02d}"
            base_date = datetime(2025, 10, 15)
            try:
                curr_date = datetime(year, month, day)
                day_offset = (curr_date - base_date).days
            except Exception:
                day_offset = 0
            return date_str, day_offset

        return str(time_val), 0

    # ---------------- MSS-physo ----------------

    def get_subject_baseline(self, subject_id, metabolite):
        """Get subject-specific baseline statistics (excluding the current measurement)."""
        if subject_id not in self.subject_history:
            return None

        history = self.subject_history[subject_id]
        entries = [(h['day_offset'], h['metabolites'][metabolite])
                   for h in history if metabolite in h['metabolites']]

        if len(entries) < 2:
            return None

        days = [e[0] for e in entries]
        values = [e[1] for e in entries]

        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 1e-6
        if std_val < 1e-9:
            std_val = 1e-6

        if len(values) >= 2:
            if len(values) >= 3:
                z = np.polyfit(days[-3:], values[-3:], 1)
            else:
                z = np.polyfit(days, values, 1)
            trend = float(z[0])
        else:
            trend = 0.0

        return {
            'mean': mean_val, 'std': std_val, 'n': len(values),
            'last_value': values[-1], 'last_day': days[-1],
            'trend': trend, 'history': values
        }

    def compute_dynamic_features(self, subject_id, metabolites, day_offset):
        """Compute dynamic features (zpop/zbase/delta/slope/cv, etc.)."""
        features = {}
        has_history = subject_id in self.subject_history and \
            len(self.subject_history[subject_id]) >= 1
        history_n = len(self.subject_history.get(subject_id, []))

        for mz in CORE_MZ:
            if mz not in metabolites:
                continue

            val = float(metabolites[mz])
            features[f'{mz}_raw'] = val

            if mz in self.population_stats:
                pop = self.population_stats[mz]
                pop_std = pop['std'] if pop['std'] > 1e-9 else 1e-6
                zpop = (val - pop['mean']) / pop_std
                features[f'{mz}_zpop'] = np.clip(zpop, -5, 5)

            baseline = self.get_subject_baseline(subject_id, mz)
            if baseline is not None and baseline['n'] >= 2:
                zbase = (val - baseline['mean']) / baseline['std']
                features[f'{mz}_zbase'] = np.clip(zbase, -5, 5)
                features[f'{mz}_delta'] = val - baseline['last_value']
                features[f'{mz}_trend'] = baseline['trend']

                day_diff = day_offset - baseline['last_day']
                if day_diff > 0:
                    features[f'{mz}_slope'] = (val - baseline['last_value']) / day_diff
                else:
                    features[f'{mz}_slope'] = 0.0

                features[f'{mz}_cv'] = baseline['std'] / (abs(baseline['mean']) + 1e-9)
                features[f'{mz}_baseline_mean'] = baseline['mean']
                features[f'{mz}_baseline_std'] = baseline['std']
            else:
                features[f'{mz}_zbase'] = features.get(f'{mz}_zpop', 0)
                features[f'{mz}_delta'] = 0.0
                features[f'{mz}_trend'] = 0.0
                features[f'{mz}_slope'] = 0.0
                features[f'{mz}_cv'] = 0.0
                features[f'{mz}_baseline_mean'] = None
                features[f'{mz}_baseline_std'] = None

        return features, has_history, history_n

    def compute_mss_dynamic(self, features, cluster_id=None):
        """Compute the dynamic MSS score: MSS = Σ(w·zbase) + cluster_bias + trend_penalty."""
        mss = 0.0
        components = {}

        for mz, weight in MSS_WEIGHTS.items():
            zbase_key = f'{mz}_zbase'
            if zbase_key in features:
                contrib = weight * features[zbase_key]
                mss += contrib
                if abs(contrib) > 0.001:
                    components[f'MSS_{mz}'] = round(contrib, 4)

        cluster_bias = 0.0
        if cluster_id and cluster_id in CLUSTER_PROFILES:
            cluster_bias = CLUSTER_PROFILES[cluster_id].get('risk_bias', 0)
            mss += cluster_bias
            components['cluster_bias'] = round(cluster_bias, 4)

        trend_penalty = 0.0
        for mz in CORE_MZ:
            trend_key = f'{mz}_trend'
            zbase_key = f'{mz}_zbase'
            if trend_key in features and zbase_key in features:
                weight = MSS_WEIGHTS.get(mz, 0)
                trend = features[trend_key]
                zbase = features[zbase_key]
                if weight > 0 and zbase > 0.5 and trend > 0:
                    trend_penalty += 0.03 * trend * weight
                elif weight < 0 and zbase < -0.5 and trend < 0:
                    trend_penalty += -0.03 * trend * weight

        mss += trend_penalty
        if abs(trend_penalty) > 0.001:
            components['trend_penalty'] = round(trend_penalty, 4)

        mss = float(np.clip(mss, -5, 5))
        return mss, components

    def assess_symptom_risk(self, features, symptom, threshold_info, has_history):
        """Assess the risk of an individual somatic symptom."""
        risk_score = 0.0
        evidence = []

        symptom_metabolites = {
            'physical_symptom': [68.0575, 84.0378, 94.0448, 120.102],
            'insomnia': [68.0575, 73.0647, 62.0302, 94.0448],
            'dizziness_palpitations': [68.0575, 120.102, 94.0448, 73.0647],
            'fidget': [68.0575, 84.0378, 55.044, 62.0302],
            'digestive_problem': [73.0647, 62.0302, 55.044, 84.0378],
            'respiratory_symptom': [68.0575, 94.0448, 120.102, 55.044],
        }

        relevant_mz = symptom_metabolites.get(symptom, CORE_MZ)

        for mz in relevant_mz:
            zbase_key = f'{mz}_zbase'
            slope_key = f'{mz}_slope'
            zpop_key = f'{mz}_zpop'

            zscore = 0
            if zbase_key in features:
                zscore = features[zbase_key]
            elif zpop_key in features:
                zscore = features[zpop_key] * 0.5

            if abs(zscore) > 1.5:
                risk_score += 0.20
                evidence.append(f"{METABOLITE_NAMES.get(mz, mz)} z={zscore:+.2f}")
            elif abs(zscore) > 1.0:
                risk_score += 0.12
            elif abs(zscore) > 0.5:
                risk_score += 0.05

            if has_history and slope_key in features:
                slope = features[slope_key]
                if abs(slope) > 0.05:
                    risk_score += 0.08
                    evidence.append(f"{METABOLITE_NAMES.get(mz, mz)} slope={slope:+.3f}")
                elif abs(slope) > 0.02:
                    risk_score += 0.04

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.50:
            level = 'HIGH'
        elif risk_score >= 0.30:
            level = 'ELEVATED'
        elif risk_score >= 0.15:
            level = 'MILD'
        else:
            level = 'NORMAL'

        return {
            'risk_score': round(risk_score, 3),
            'risk_level': level,
            'threshold': threshold_info['threshold'],
            'evidence': evidence[:3]
        }

    # ---------------- MSS-psycho----------------

    def compute_psycho_scores(self, subject_id, metabolites_now, questionnaire_now=None):
        """
        Compute MSS-psycho psychological state scores and alerts.

        Prediction chain: predicted = base + Σ w·Δmz
          - base  = the subject's most recent questionnaire score (when questionnaire history exists)
                    or the population questionnaire mean (when no questionnaire history exists; marked LOW_CONFIDENCE)
          - Δmz   = current abundance − previous abundance

        Args:
            subject_id: Subject ID
            metabolites_now: Current metabolite dict {mz: value}
            questionnaire_now: Current measured questionnaire score dict (optional; used only for recording comparisons)

        Returns:
            dict {state: {...}} or None (no previous measurement; cannot be computed)
        """
        history = self.subject_history.get(subject_id, [])
        if len(history) < 1:
            return None  # first measurement, no Δmz

        prev_metabolites = history[-1]['metabolites']

        # Find the most recent valid questionnaire score
        last_scores = {}
        for h in reversed(history):
            qs = h.get('questionnaire') or {}
            for state, model in PSYCHO_MODELS.items():
                col = model['questionnaire_col']
                if state not in last_scores and qs.get(col) is not None:
                    last_scores[state] = float(qs[col])
            if len(last_scores) == len(PSYCHO_MODELS):
                break

        results = {}
        for state, model in PSYCHO_MODELS.items():
            thr = model['threshold']
            cmp_op = model['cmp']
            resid_sd = model['resid_sd']

            # Δmz and predicted change amount
            delta_pred = 0.0
            drivers = []
            n_matched = 0
            for mz, w in model['weights'].items():
                if mz in metabolites_now and mz in prev_metabolites:
                    dmz = float(metabolites_now[mz]) - float(prev_metabolites[mz])
                    contrib = w * dmz
                    delta_pred += contrib
                    n_matched += 1
                    if abs(contrib) > 0.05 * resid_sd:
                        drivers.append({
                            'mz': mz,
                            'name': METABOLITE_NAMES.get(mz, str(mz)),
                            'delta': round(dmz, 5),
                            'contribution': round(contrib, 3),
                            'direction': '↑' if dmz > 0 else '↓',
                        })

            if n_matched == 0:
                results[state] = {'status': 'NO_DATA'}
                continue

            # Baseline score: individual questionnaire history takes precedence; otherwise, the population mean (low confidence)
            if state in last_scores:
                base = last_scores[state]
                base_source = 'subject_history'
                low_conf = False
            else:
                base = PSYCHO_POP_STATS[state]['mean']
                base_source = 'population_mean'
                low_conf = True

            predicted = base + delta_pred

            # Whether clinical thresholds are crossed
            def _cross(v):
                return v >= thr if cmp_op == '>=' else v > thr

            crossing = _cross(predicted)
            near_thr = _cross(predicted + 0.5 * resid_sd)  #  < 0.5 resid_sd
            worsening = delta_pred >= 0.5 * resid_sd       # Predict significant deterioration



            if crossing:
                level = 'HIGH'
            elif near_thr or worsening:
                level = 'ELEVATED'
            elif _cross(predicted + resid_sd) or delta_pred > 0:
                level = 'MILD'
            else:
                level = 'NORMAL'

            # Normalized risk score (0–1, for radar chart): full scale = threshold + 1 resid_sd
            full_scale = thr + resid_sd
            risk_score = float(np.clip(predicted / full_scale, 0, 1))

            drivers.sort(key=lambda x: abs(x['contribution']), reverse=True)

            results[state] = {
                'status': ('LOW_CONFIDENCE' if (low_conf or model['confidence'] == 'LOW')
                           else 'MEDIUM_CONFIDENCE' if model['confidence'] == 'MEDIUM'
                           else 'OK'),
                'label': model['label'],
                'predicted_score': round(predicted, 2),
                'delta_pred': round(delta_pred, 2),
                'base_score': round(base, 2),
                'base_source': base_source,
                'threshold': thr,
                'threshold_desc': model['threshold_desc'],
                'crossing_threshold': bool(crossing),
                'risk_level': level,
                'risk_score': round(risk_score, 3),
                'model_confidence': model['confidence'],
                'loso_auc': model['loso_auc'],
                'drivers': drivers[:4],
                'measured_now': (float(questionnaire_now[model['questionnaire_col']])
                                 if questionnaire_now and
                                 questionnaire_now.get(model['questionnaire_col']) is not None else None),
            }

        return results

    # ---------------- Comprehensive assessment ----------------

    def _trend_counts(self, features):
        """Count worsening/improving/stable core metabolites (using the same criteria as _get_trend_summary)."""
        worsening = improving = stable = total = 0
        for mz in CORE_MZ:
            zbase_key = f'{mz}_zbase'
            slope_key = f'{mz}_slope'
            if zbase_key in features and slope_key in features:
                total += 1
                zbase = features[zbase_key]
                slope = features[slope_key]
                if abs(zbase) > 0.5 and ((zbase > 0 and slope > 0) or (zbase < 0 and slope < 0)):
                    worsening += 1
                elif abs(zbase) > 0.5 and ((zbase > 0 and slope < 0) or (zbase < 0 and slope > 0)):
                    improving += 1
                else:
                    stable += 1
        return worsening, improving, stable, total

    def determine_overall_risk(self, mss_score, symptom_risks, has_history, cluster_id,
                               psycho_results=None, n_history=0, n_deteriorating=0):
        """
        Determine the overall risk level (GREEN / YELLOW / RED / BLUE, stratified by measurement stage).

        Design principle: avoid excessive alerts during the baseline period; during the dynamic period, RED corresponds only to extreme metabolic deterioration.

        - BLUE (early warning):
            * n_history == 0 (single-point baseline assessment): flag only for severe deviation from population references
              (MSS < -1.5 or any symptom risk >= 0.5), with a recommendation to repeat the measurement promptly for confirmation;
            * n_history == 1 (baseline construction): individual standard deviation is not yet robust; do not trigger dynamic
              RED, always flag BLUE, and recommend repeat measurement within 24-48 h;
        - RED (dynamic alert, n_history >= 2): only when extreme criteria are met—
          profound MSS deviation (< -1.5), or >=6/7 core metabolites worsening and peak symptom risk >= 0.6;
        - GREEN (low risk): <=2 worsening metabolites and peak symptom risk <= 0.20;
        - All other cases are YELLOW.
        Psychological states (MSS-psycho) are displayed independently in the report and are not incorporated into the overall level
        (see the psychological branch panel for psychological threshold crossings, preventing a single dimension from dominating the overall assessment).
        """
        max_symptom_risk = max([r['risk_score'] for r in symptom_risks.values()]) if symptom_risks else 0
        elevated_symptoms = [s for s, r in symptom_risks.items() if r['risk_level'] == 'ELEVATED']

        # ---- Stage 1: Baseline phase----
        if n_history == 0:
            if mss_score < -1.5 or max_symptom_risk >= 0.5:
                return 'BLUE', ('EARLY WARNING (baseline assessment): severe deviation vs population '
                                'norms, not confirmable without individual baseline; '
                                'repeat measurement within 24-48 h')
            elif mss_score < -0.5 or max_symptom_risk >= 0.3 or len(elevated_symptoms) >= 1:
                return 'YELLOW', 'MONITOR CLOSELY (baseline assessment)'
            else:
                return 'GREEN', 'NORMAL (baseline assessment)'

        if n_history == 1:
            return 'BLUE', ('EARLY WARNING (baseline under construction): individual SD not robust, '
                            'no dynamic RED issued; repeat measurement within 24-48 h')

        # ---- Phase 2: Fully dynamic mode ----
        if mss_score < -1.5 or (n_deteriorating >= 6 and max_symptom_risk >= 0.6):
            return 'RED', 'IMMEDIATE ATTENTION REQUIRED'

        if n_deteriorating <= 2 and max_symptom_risk <= 0.20:
            return 'GREEN', 'NORMAL'

        return 'YELLOW', 'MONITOR CLOSELY'

    def evaluate(self, row_dict):
        """
        Evaluate a single measurement.

        Args:
            row_dict: Single-row data dictionary; must contain SubjectID, Time, and metabolite m/z columns;
                      optional separate questionnaire columns: fatigue_total / emotion_total / stress_total

        Returns:
            result_dict: Complete evaluation result (including the psycho_scores field)
        """
        subject_id = int(row_dict.get('SubjectID') or row_dict.get('subject_id'))
        time_val = row_dict.get('Time') or row_dict.get('time')

        date_str, day_offset = self.parse_timestamp(time_val)

        # Extract metabolites (CORE_MZ ∪ PSYCHO_MZ)
        all_mz = sorted(set(CORE_MZ) | set(PSYCHO_MZ))
        metabolites = {}
        for mz in all_mz:
            for key in (mz, str(mz), str(int(mz)) if float(mz).is_integer() else None):
                if key is not None and key in row_dict and pd.notna(row_dict[key]):
                    metabolites[mz] = float(row_dict[key])
                    break

        # Extract current questionnaire score (optional)
        questionnaire_now = {}
        for model in PSYCHO_MODELS.values():
            col = model['questionnaire_col']
            v = row_dict.get(col)
            if v is not None and pd.notna(v):
                questionnaire_now[col] = float(v)
        if not questionnaire_now:
            questionnaire_now = None

        cluster_id = self.subject_clusters.get(subject_id, None)

        # MSS-physo
        features, has_history, history_n = self.compute_dynamic_features(
            subject_id, metabolites, day_offset
        )
        mss_score, mss_components = self.compute_mss_dynamic(features, cluster_id)

        symptom_risks = {}
        for symptom, info in CLINICAL_THRESHOLDS.items():
            symptom_risks[symptom] = self.assess_symptom_risk(features, symptom, info, has_history)

        # MSS-psycho
        psycho_results = self.compute_psycho_scores(subject_id, metabolites, questionnaire_now)

        # Overall risk (four-level assessment, v3.1)
        n_deteriorating = self._trend_counts(features)[0] if has_history else 0
        overall_risk, alert_msg = self.determine_overall_risk(
            mss_score, symptom_risks, has_history, cluster_id, psycho_results,
            n_history=history_n, n_deteriorating=n_deteriorating
        )

        result = {
            'subject_id': subject_id,
            'timestamp_raw': time_val,
            'timestamp_actual': date_str,
            'day_offset': day_offset,
            'has_history': has_history,
            'n_history': history_n,
            'cluster_id': cluster_id,
            'cluster_name': CLUSTER_PROFILES.get(cluster_id, {}).get('description', 'Unknown') if cluster_id else 'Unknown',

            'mss_score': round(mss_score, 4),
            'mss_components': mss_components,
            'overall_risk': overall_risk,
            'alert_message': alert_msg,
            'early_warning': overall_risk == 'BLUE',
            'n_deteriorating': n_deteriorating,

            'symptom_risks': symptom_risks,
            'psycho_scores': psycho_results,

            'dynamic_features': features,
            'abnormal_metabolites': self._get_abnormal_metabolites(features, has_history),
            'trend_summary': self._get_trend_summary(features, has_history),
        }

        self._store_measurement(subject_id, date_str, day_offset, metabolites,
                                result, questionnaire_now)

        return result

    def _get_abnormal_metabolites(self, features, has_history):
        """Extract the list of abnormal metabolites."""
        abnormal = []
        for mz in CORE_MZ:
            zbase_key = f'{mz}_zbase'
            zpop_key = f'{mz}_zpop'

            if zbase_key in features:
                zscore = features[zbase_key]
                basis = 'baseline' if has_history else 'population'
            elif zpop_key in features:
                zscore = features[zpop_key]
                basis = 'population'
            else:
                continue

            if abs(zscore) > 1.0:
                abnormal.append({
                    'mz': mz,
                    'name': METABOLITE_NAMES.get(mz, str(mz)),
                    'zscore': round(zscore, 3),
                    'direction': '↑' if zscore > 0 else '↓',
                    'severity': 'SEVERE' if abs(zscore) > 2.5 else 'MODERATE' if abs(zscore) > 1.5 else 'MILD',
                    'basis': basis
                })

        abnormal.sort(key=lambda x: abs(x['zscore']), reverse=True)
        return abnormal[:5]

    def _get_trend_summary(self, features, has_history):
        """Get the trend summary."""
        if not has_history:
            return "No prior measurements. Baseline assessment using population norms."

        worsening, improving, stable, total = self._trend_counts(features)
        if total == 0:
            return "Insufficient data for trend analysis."

        if worsening >= max(2, total // 3):
            return f"DETERIORATING: {worsening}/{total} metabolites showing adverse trends"
        elif improving >= max(2, total // 3):
            return f"IMPROVING: {improving}/{total} metabolites showing favorable trends"
        else:
            return f"STABLE: {stable}/{total} metabolites within normal variation"

    def _store_measurement(self, subject_id, date_str, day_offset, metabolites,
                           result, questionnaire_now=None):
        """Store the measurement in history (including questionnaire scores for use as MSS-psycho baselines)."""
        if subject_id not in self.subject_history:
            self.subject_history[subject_id] = []

        self.subject_history[subject_id].append({
            'date': date_str,
            'day_offset': day_offset,
            'metabolites': metabolites,
            'questionnaire': questionnaire_now,
            'mss_score': result['mss_score'],
            'overall_risk': result['overall_risk']
        })

        if len(self.subject_history[subject_id]) > 10:
            self.subject_history[subject_id] = self.subject_history[subject_id][-10:]

    # ---------------- report ----------------

    def generate_report(self, result):
        """Generate a human-readable alert report (somatic and psychological dimensions)."""
        lines = []
        lines.append("=" * 72)
        lines.append("MAREHEALTH-DYNAMIC MSS ALERT REPORT (v3.1: PHY-SO + PSYCHO)")
        lines.append("=" * 72)
        lines.append(f"Subject ID:        {result['subject_id']}")
        lines.append(f"Detection Time:    {result['timestamp_actual']} (Day {result['day_offset']})")
        lines.append(f"History Status:    {result['n_history']} prior measurement(s)")
        lines.append(f"Metabolic Cluster: {result['cluster_name']} (ID: {result['cluster_id']})")
        lines.append("")

        risk = result['overall_risk']
        risk_symbol = {"GREEN": "✓ NORMAL", "YELLOW": "⚠ CAUTION", "RED": "✗ ALERT",
                       "BLUE": "◆ EARLY WARNING"}
        lines.append(f"OVERALL RISK:      {risk} — {risk_symbol.get(risk, '')}")
        lines.append(f"Alert Message:     {result['alert_message']}")
        lines.append("")

        # MSS-physo
        lines.append("[MSS-physo] Somatic Symptom Assessment")
        lines.append(f"Dynamic MSS Score: {result['mss_score']:.4f}")
        if result['mss_components']:
            lines.append("  Components:")
            for comp, val in result['mss_components'].items():
                if abs(val) > 0.001:
                    sign = "+" if val >= 0 else ""
                    lines.append(f"    {comp:20s}: {sign}{val:.4f}")
        lines.append("")
        lines.append(f"{'Symptom':<30s} | {'Level':<10s} | {'Score':<6s} | Evidence")
        lines.append("-" * 72)
        for symptom, risk_info in result['symptom_risks'].items():
            sym_name = CLINICAL_THRESHOLDS.get(symptom, {}).get('symptom', symptom)
            ev_str = "; ".join(risk_info['evidence'][:2]) if risk_info['evidence'] else "-"
            lines.append(f"{sym_name:<30s} | {risk_info['risk_level']:<10s} | {risk_info['risk_score']:.3f}  | {ev_str}")
        lines.append("")

        # MSS-psycho
        lines.append("[MSS-psycho] Psychological State Assessment")
        lines.append("-" * 72)
        if result['psycho_scores'] is None:
            lines.append("  INSUFFICIENT DATA: first measurement, no previous spectrum")
            lines.append("  available for delta computation. Psycho assessment starts")
            lines.append("  from the 2nd measurement.")
        else:
            lines.append(f"{'State':<10s} | {'Pred':>6s} | {'Δpred':>6s} | {'Thr':>4s} | {'Level':<9s} | Note")
            lines.append("-" * 72)
            for state, pr in result['psycho_scores'].items():
                if pr.get('status') == 'NO_DATA':
                    lines.append(f"{pr if isinstance(pr, str) else state:<10s} | NO MATCHED METABOLITES")
                    continue
                note = []
                if pr['status'] == 'LOW_CONFIDENCE':
                    note.append('low-conf')
                elif pr['status'] == 'MEDIUM_CONFIDENCE':
                    note.append('med-conf')
                if pr['crossing_threshold']:
                    note.append('CROSSING')
                if pr['measured_now'] is not None:
                    note.append(f"meas={pr['measured_now']:.0f}")
                lines.append(f"{pr['label']:<10s} | {pr['predicted_score']:>6.1f} | "
                             f"{pr['delta_pred']:>+6.1f} | {pr['threshold']:>4d} | "
                             f"{pr['risk_level']:<9s} | {', '.join(note)}")
            lines.append("")
            # driver metabolites
            for state, pr in result['psycho_scores'].items():
                if pr.get('drivers'):
                    drv = "; ".join(f"{d['name']}({d['direction']}, {d['contribution']:+.1f})"
                                    for d in pr['drivers'][:3])
                    lines.append(f"  {pr['label']} drivers: {drv}")
        lines.append("")

        # abnormal metabolites
        if result['abnormal_metabolites']:
            lines.append("Abnormal Metabolites (MSS core):")
            lines.append("-" * 72)
            lines.append(f"{'Direction':<4s} {'Name':<25s} {'m/z':<10s} {'z-score':<10s} {'Severity':<10s} {'Basis'}")
            lines.append("-" * 72)
            for m in result['abnormal_metabolites']:
                basis_tag = f"(vs {m['basis']})"
                lines.append(f"{m['direction']:<4s} {m['name']:<25s} {m['mz']:<10.4f} {m['zscore']:>+8.2f}   {m['severity']:<10s} {basis_tag}")
        else:
            lines.append("All monitored metabolites within normal range.")
        lines.append("")

        lines.append(f"Trend Summary:     {result['trend_summary']}")
        lines.append("")

        # suggestion
        lines.append("Recommendations:")
        psycho = result['psycho_scores'] or {}
        psycho_high = [s for s, pr in psycho.items() if pr.get('risk_level') == 'HIGH']
        if risk == 'RED':
            lines.append("  • IMMEDIATE: Contact medical officer for evaluation")
            lines.append("  • Schedule follow-up measurement within 24 hours")
            lines.append("  • Review workload and rest schedule immediately")
            lines.append("  • Consider temporary duty restriction")
            if psycho_high:
                lines.append(f"  • Psychological states crossing clinical threshold: "
                             f"{', '.join(s.upper() for s in psycho_high)} — consider "
                             f"psychological counseling / workload relief")
        elif risk == 'BLUE':
            lines.append("  • EARLY WARNING: Baseline under construction, result for reference only")
            lines.append("  • Recommend completing >=3 measurements before formal assessment")
            lines.append("  • Continue routine monitoring, no special intervention needed")
            lines.append("  • Repeat measurement within 24-48 h to establish robust baseline")
        elif risk == 'YELLOW':
            lines.append("  • MONITOR: Increase measurement frequency to daily")
            lines.append("  • Review sleep quality and stress levels")
            lines.append("  • Consider workload adjustment if symptoms persist")
            lines.append("  • Schedule follow-up within 48-72 hours")
            if psycho:
                el = [s for s, pr in psycho.items() if pr.get('risk_level') == 'ELEVATED']
                if el:
                    lines.append(f"  • Watch psychological states: {', '.join(s.upper() for s in el)}")
        else:
            lines.append("  • CONTINUE: Maintain current monitoring schedule")
            lines.append("  • Routine health maintenance")
            lines.append("  • Next measurement as per standard protocol")
        lines.append("=" * 72)

        return "\n".join(lines)

    def evaluate_csv(self, filepath):
        """
        Evaluate all samples in a CSV/Excel file.
        Sort by SubjectID and Time to ensure that each subject's measurements are processed chronologically.
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df = df.sort_values(['SubjectID', 'Time']).reset_index(drop=True)

        results = []
        for _, row in df.iterrows():
            result = self.evaluate(row.to_dict())
            results.append(result)

        flat_results = []
        for r in results:
            flat = {
                'subject_id': r['subject_id'],
                'timestamp_raw': r['timestamp_raw'],
                'timestamp_actual': r['timestamp_actual'],
                'day_offset': r['day_offset'],
                'has_history': r['has_history'],
                'n_history': r['n_history'],
                'cluster_id': r['cluster_id'],
                'cluster_name': r['cluster_name'],
                'mss_score': r['mss_score'],
                'overall_risk': r['overall_risk'],
                'alert_message': r['alert_message'],
                'trend_summary': r['trend_summary'],
            }
            for symptom, risk_info in r['symptom_risks'].items():
                flat[f'{symptom}_risk'] = risk_info['risk_level']
                flat[f'{symptom}_score'] = risk_info['risk_score']
            # flattening
            if r['psycho_scores']:
                for state, pr in r['psycho_scores'].items():
                    if pr.get('status') == 'NO_DATA':
                        continue
                    flat[f'psycho_{state}_pred'] = pr['predicted_score']
                    flat[f'psycho_{state}_delta'] = pr['delta_pred']
                    flat[f'psycho_{state}_risk'] = pr['risk_level']
                    flat[f'psycho_{state}_score'] = pr['risk_score']
                    flat[f'psycho_{state}_crossing'] = pr['crossing_threshold']
                    flat[f'psycho_{state}_status'] = pr['status']
                    flat[f'psycho_{state}_measured'] = pr['measured_now']
            flat_results.append(flat)

        return pd.DataFrame(flat_results)


# ==================== Demo ====================

def demo():
    """Demonstrate the v3.0 dynamic alert system (somatic and psychological dimensions)."""
    print("=" * 72)
    print("MareHealth-dynamic MSS Alert System v3.1 - Demo (MSS-physo + MSS-psycho)")
    print("=" * 72)

    system = MareHealthMSSAlertSystem()

    cluster_assignments = {
        1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 6: 3, 7: 1, 8: 2, 9: 2, 10: 2,
        11: 2, 12: 3, 13: 2, 14: 1, 15: 3, 16: 2, 17: 1, 18: 2, 19: 2, 20: 3,
        21: 3, 22: 3, 23: 2, 24: 2, 25: 1, 26: 1, 27: 1, 28: 3, 29: 3, 30: 2,
        31: 2, 32: 3, 33: 3, 34: 2, 35: 3, 36: 3, 37: 2, 38: 3, 39: 3
    }
    for sid, cid in cluster_assignments.items():
        system.set_subject_cluster(sid, cid)

    df = pd.read_excel('/mnt/agents/upload/final_merged_data_Top20.xlsx')
    df = df.sort_values(['SubjectID', 'Time']).reset_index(drop=True)

    #  Subject 7(acute stress-sensitive subtype) full time series
    print("\n[DEMO] Subject 7 (High-Stress Responder) - Time-Series:")
    print("-" * 72)

    subject_data = df[df['SubjectID'] == 7].copy()
    for _, row in subject_data.iterrows():
        result = system.evaluate(row.to_dict())
        psy = result['psycho_scores']
        psy_str = " | ".join(
            f"{p['label']}={p['predicted_score']:.1f}({p['risk_level']})"
            for p in psy.values() if isinstance(p, dict) and 'predicted_score' in p
        ) if psy else "psycho: N/A (first measurement)"
        print(f"{result['timestamp_actual']} Day{result['day_offset']:>2d} "
              f"[h={result['n_history']}] MSS={result['mss_score']:+.3f} "
              f"Risk={result['overall_risk']:<6s} | {psy_str}")

    # Detailed report of the last measurement
    print("\n" + "=" * 72)
    print("[DEMO] Detailed Report - Subject 7, Last Measurement:")
    print("=" * 72)
    last_result = system.evaluate(subject_data.iloc[-1].to_dict())
    print(system.generate_report(last_result))

    # Batch processing
    print("\n[DEMO] Batch Processing All Subjects...")
    system2 = MareHealthMSSAlertSystem()
    for sid, cid in cluster_assignments.items():
        system2.set_subject_cluster(sid, cid)

    results_df = system2.evaluate_csv('/mnt/agents/upload/final_merged_data_Top20.xlsx')

    print(f"Processed {len(results_df)} measurements from "
          f"{results_df['subject_id'].nunique()} subjects")
    print(f"\nRisk Distribution:")
    print(results_df['overall_risk'].value_counts())
    print(f"\nPsycho HIGH counts:")
    for state in ['fatigue', 'emotion', 'stress']:
        col = f'psycho_{state}_risk'
        if col in results_df.columns:
            print(f"  {state}: {(results_df[col] == 'HIGH').sum()} HIGH / "
                  f"{results_df[col].notna().sum()} evaluated")

    output_csv = '/mnt/agents/output/marehealth_dynamic_alert_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n[SAVE] Results saved to: {output_csv}")

    report_path = '/mnt/agents/output/marehealth_dynamic_alert_reports.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        system3 = MareHealthMSSAlertSystem()
        for sid, cid in cluster_assignments.items():
            system3.set_subject_cluster(sid, cid)
        for _, row in df.iterrows():
            r = system3.evaluate(row.to_dict())
            f.write(system3.generate_report(r))
            f.write("\n\n")
    print(f"[SAVE] Reports saved to: {report_path}")

    return results_df


if __name__ == '__main__':
    demo()
