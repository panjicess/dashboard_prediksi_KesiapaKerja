import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kesiapan Kerja Mahasiswa",
    page_icon="📊",
    layout="wide"
)

# CSS Kustom
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.5rem 0; margin-bottom: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; color: white; }
    .main-header p { font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.9; color: white; }
    .prediction-box { padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .prediction-box.siap { background: linear-gradient(135deg, #d4edda, #b7e4c7); border: 2px solid #28a745; }
    .prediction-box.belum { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 2px solid #dc3545; }
    .prediction-box h2 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .prediction-box .probability { font-size: 1.2rem; color: #555; margin-top: 0.5rem; }
    .metric-card { background: white; padding: 1.2rem; border-radius: 10px; text-align: center; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .metric-card .label { font-size: 0.9rem; color: #888; margin-top: 0.3rem; }
    .recommendation-box { padding: 0.8rem 1rem; border-radius: 8px; margin: 0.4rem 0; font-size: 0.95rem; }
    .recommendation-box.improve { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .recommendation-box.maintain { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
    .section-title { color: #1a1a2e; font-weight: 600; font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e9ecef; }
    .footer { text-align: center; padding: 2rem 0; color: #888; font-size: 0.85rem; border-top: 1px solid #e9ecef; margin-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #f0f2f6; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #667eea; color: white; }
    .stButton button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; transition: all 0.3s; }
    .stButton button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
    .stDownloadButton button { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; font-weight: 600; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; transition: all 0.3s; }
    .stDownloadButton button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4); }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="main-header">
    <h1>Prediksi Kesiapan Kerja Mahasiswa</h1>
    <p>Berbasis Soft Skills, Life Skills, dan Technical Skills</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL DAN AMBIL NAMA FITUR
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    st.success("Model berhasil dimuat")
    
    # AMBIL NAMA FITUR DARI MODEL (PERSIS SESUAI URUTAN)
    all_attrs = list(scaler.feature_names_in_)
    st.info(f"Model menggunakan {len(all_attrs)} fitur")
    
    # TAMPILKAN NAMA FITUR UNTUK VERIFIKASI
    with st.expander("Lihat daftar fitur dari model (urutan)"):
        st.write(all_attrs)
    
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file 'random_forest_model.pkl' dan 'scaler.pkl' berada di folder yang sama.")
    st.stop()

# ============================================
# BAGI FITUR PER KELOMPOK
# ============================================
soft_attrs = []
life_attrs = []
tech_attrs = []

for col in all_attrs:
    col_lower = col.lower()
    if 'ratesoft' in col_lower or 'decision' in col_lower or 'planning' in col_lower or 'teamwork' in col_lower or 'confident' in col_lower or 'meet' in col_lower or 'stress' in col_lower or 'communicate' in col_lower or 'boosting' in col_lower or 'multitask' in col_lower or 'working' in col_lower or 'opinion' in col_lower or 'resolve' in col_lower or 'convey' in col_lower or 'alone' in col_lower or 'rearranging' in col_lower or 'inspiration' in col_lower or 'motivation' in col_lower or 'emotional' in col_lower:
        soft_attrs.append(col)
    elif 'ratelife' in col_lower or 'demonstrator' in col_lower or 'good leader' in col_lower or 'good listener' in col_lower or 'oral' in col_lower or 'team building' in col_lower or 'area of interest' in col_lower or 'good presenter' in col_lower or 'coach' in col_lower or 'interpersonal' in col_lower:
        life_attrs.append(col)
    elif 'ratetech' in col_lower or 'time management' in col_lower or 'quality management' in col_lower or 'blend' in col_lower or 'gain' in col_lower or 'new hands-on' in col_lower or 'group project' in col_lower or 'developing projects' in col_lower or 'cost/time' in col_lower or 'administrative' in col_lower:
        tech_attrs.append(col)
    else:
        soft_attrs.append(col)

# ============================================
# FUNGSI TEMPLATE
# ============================================
def create_template():
    df_template = pd.DataFrame(columns=all_attrs + ['Nama_Mahasiswa'])
    df_template.loc[0] = [2]*len(all_attrs) + ['Contoh Mahasiswa']
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, sheet_name='Template', index=False)
    return output.getvalue()

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict_single(input_data):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[all_attrs]
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return prediction, probability

def predict_batch(df):
    df = df[all_attrs]
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    return predictions, probabilities

def get_feature_importance():
    importance_df = pd.DataFrame({
        'feature': all_attrs,
        'importance': model.feature_importances_
    })
    
    soft_avg = importance_df[importance_df['feature'].isin(soft_attrs)]['importance'].mean()
    life_avg = importance_df[importance_df['feature'].isin(life_attrs)]['importance'].mean()
    tech_avg = importance_df[importance_df['feature'].isin(tech_attrs)]['importance'].mean()
    
    total_avg = soft_avg + life_avg + tech_avg
    
    return {
        'Soft Skills': soft_avg / total_avg * 100,
        'Life Skills': life_avg / total_avg * 100,
        'Technical Skills': tech_avg / total_avg * 100
    }

# ============================================
# TAB INPUT
# ============================================
tab1, tab2 = st.tabs(["Input Manual", "Input Excel"])

# ============================================
# TAB 1: INPUT MANUAL
# ============================================
with tab1:
    st.markdown('<p class="section-title">Input Data Mahasiswa</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with col1:
        st.subheader(f"Soft Skills ({len(soft_attrs)} atribut)")
        for attr in soft_attrs:
            label = attr.replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').title()
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"soft_{attr}")
    
    with col2:
        st.subheader(f"Life Skills ({len(life_attrs)} atribut)")
        for attr in life_attrs:
            label = attr.replace('-', ' ').replace('ratelife', 'Rate Life').title()
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"life_{attr}")
        
        st.subheader(f"Technical Skills ({len(tech_attrs)} atribut)")
        for attr in tech_attrs:
            label = attr.replace('-', ' ').replace('ratetech', 'Rate Tech').title()
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"tech_{attr}")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_manual = st.button("Prediksi Kesiapan Kerja", type="primary", use_container_width=True)
    
    if predict_manual:
        with st.spinner("Memproses data..."):
            prediction, probability = predict_single(input_data)
            contributions = get_feature_importance()
            soft_score = np.mean([input_data[attr] for attr in soft_attrs])
            life_score = np.mean([input_data[attr] for attr in life_attrs])
            tech_score = np.mean([input_data[attr] for attr in tech_attrs])
            
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box siap">
                    <h2 style="color: #155724;">SIAP KERJA</h2>
                    <div class="probability">Probabilitas: {probability*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box belum">
                    <h2 style="color: #721c24;">BELUM SIAP KERJA</h2>
                    <div class="probability">Probabilitas: {probability*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{soft_score:.2f}</div>
                    <div class="label">Soft Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{life_score:.2f}</div>
                    <div class="label">Life Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{tech_score:.2f}</div>
                    <div class="label">Technical Skills</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown('<p class="section-title">Radar Chart Skor per Kelompok</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))
                categories = ['Soft Skills', 'Life Skills', 'Technical Skills']
                values = [soft_score, life_score, tech_score]
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
                ax.fill(angles, values, alpha=0.25, color='#667eea')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 5)
                ax.set_title('Skor per Kelompok', fontsize=12)
                st.pyplot(fig)
                plt.close()
            
            with col_g2:
                st.markdown('<p class="section-title">Kontribusi Faktor (Rata-rata per Fitur)</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                categories = list(contributions.keys())
                values = list(contributions.values())
                colors = ['#667eea', '#764ba2', '#f093fb']
                bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
                ax.set_ylim(0, 100)
                ax.set_ylabel('Kontribusi (%)')
                ax.set_title('Feature Importance per Kelompok', fontsize=12)
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            st.markdown('<p class="section-title">Rekomendasi Pengembangan</p>', unsafe_allow_html=True)
            
            all_scores = {attr: input_data[attr] for attr in all_attrs}
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1])
            low_attrs = [attr for attr, score in sorted_scores if score <= 2]
            high_attrs = [attr for attr, score in sorted_scores if score >= 4]
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("**Perlu Ditingkatkan (Skor <= 2)**")
                if low_attrs:
                    for attr in low_attrs[:10]:
                        label = attr.replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').replace('ratelife', 'Rate Life').replace('ratetech', 'Rate Tech').title()
                        st.markdown(f"""
                        <div class="recommendation-box improve">
                            {label}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Semua atribut sudah baik. Pertahankan!")
            
            with col_rec2:
                st.markdown("**Perlu Dipertahankan (Skor >= 4)**")
                if high_attrs:
                    for attr in high_attrs[:10]:
                        label = attr.replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').replace('ratelife', 'Rate Life').replace('ratetech', 'Rate Tech').title()
                        st.markdown(f"""
                        <div class="recommendation-box maintain">
                            {label}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Belum ada atribut dengan skor tinggi. Fokus pada peningkatan.")
            
            st.markdown('<p class="section-title">Detail Skor per Atribut</p>', unsafe_allow_html=True)
            tab_soft, tab_life, tab_tech = st.tabs(["Soft Skills", "Life Skills", "Technical Skills"])
            
            with tab_soft:
                df_soft = pd.DataFrame({
                    'Atribut': [a.replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').title() for a in soft_attrs],
                    'Skor': [input_data[attr] for attr in soft_attrs]
                })
                df_soft['Skor'] = df_soft['Skor'].astype(int)
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#28a745' if s >= 4 else '#ffc107' if s >= 3 else '#dc3545' for s in df_soft['Skor']]
                bars = ax.barh(df_soft['Atribut'], df_soft['Skor'], color=colors)
                ax.set_xlim(0, 5)
                ax.set_xlabel('Skor')
                ax.set_title('Soft Skills')
                for bar, val in zip(bars, df_soft['Skor']):
                    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')
                st.pyplot(fig)
                plt.close()
            
            with tab_life:
                df_life = pd.DataFrame({
                    'Atribut': [a.replace('-', ' ').replace('ratelife', 'Rate Life').title() for a in life_attrs],
                    'Skor': [input_data[attr] for attr in life_attrs]
                })
                df_life['Skor'] = df_life['Skor'].astype(int)
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#28a745' if s >= 4 else '#ffc107' if s >= 3 else '#dc3545' for s in df_life['Skor']]
                bars = ax.barh(df_life['Atribut'], df_life['Skor'], color=colors)
                ax.set_xlim(0, 5)
                ax.set_xlabel('Skor')
                ax.set_title('Life Skills')
                for bar, val in zip(bars, df_life['Skor']):
                    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')
                st.pyplot(fig)
                plt.close()
            
            with tab_tech:
                df_tech = pd.DataFrame({
                    'Atribut': [a.replace('-', ' ').replace('ratetech', 'Rate Tech').title() for a in tech_attrs],
                    'Skor': [input_data[attr] for attr in tech_attrs]
                })
                df_tech['Skor'] = df_tech['Skor'].astype(int)
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#28a745' if s >= 4 else '#ffc107' if s >= 3 else '#dc3545' for s in df_tech['Skor']]
                bars = ax.barh(df_tech['Atribut'], df_tech['Skor'], color=colors)
                ax.set_xlim(0, 5)
                ax.set_xlabel('Skor')
                ax.set_title('Technical Skills')
                for bar, val in zip(bars, df_tech['Skor']):
                    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')
                st.pyplot(fig)
                plt.close()

# ============================================
# TAB 2: INPUT EXCEL
# ============================================
with tab2:
    st.markdown('<p class="section-title">Upload Data Excel</p>', unsafe_allow_html=True)
    
    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    with col_t2:
        template_data = create_template()
        st.download_button(
            label="Download Template Excel",
            data=template_data,
            file_name="template_prediksi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("""
    **Petunjuk:**
    1. Download template Excel di atas
    2. Isi data mahasiswa (skala 0-5) pada kolom yang tersedia
    3. Upload file yang sudah diisi
    """)
    
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_excel(uploaded_file)
            missing_cols = [col for col in all_attrs if col not in df_batch.columns]
            if missing_cols:
                st.error(f"Kolom tidak ditemukan: {missing_cols[:5]}...")
            else:
                st.success(f"Data berhasil dimuat: {len(df_batch)} responden")
                
                if st.button("Prediksi Semua Data", type="primary", use_container_width=True):
                    with st.spinner("Memproses data..."):
                        predictions, probabilities = predict_batch(df_batch)
                        df_batch['Prediksi'] = ['Siap Kerja' if p == 1 else 'Belum Siap' for p in predictions]
                        df_batch['Probabilitas'] = probabilities
                        
                        # C. HASIL PREDIKSI KESIAPAN KERJA
                        st.markdown('<p class="section-title">C. Hasil Prediksi Kesiapan Kerja</p>', unsafe_allow_html=True)
                        
                        total = len(df_batch)
                        siap = df_batch[df_batch['Prediksi'] == 'Siap Kerja'].shape[0]
                        belum = df_batch[df_batch['Prediksi'] == 'Belum Siap'].shape[0]
                        
                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.metric("Total Responden", total)
                        with col_c2:
                            st.metric("Siap Kerja", f"{siap} ({siap/total*100:.1f}%)")
                        with col_c3:
                            st.metric("Belum Siap", f"{belum} ({belum/total*100:.1f}%)")
                        
                        # Perbandingan dengan dataset publik
                        publik_siap = 28.8
                        publik_belum = 71.2
                        privat_siap = siap/total*100
                        privat_belum = belum/total*100
                        
                        st.markdown("**Perbandingan dengan Dataset Publik (Mendeley):**")
                        
                        # PERBAIKAN ADA DI SINI
                        col_comp1, col_comp2, col_comp3 = st.columns(3)
                        with col_comp1:
                            st.metric("Dataset", "Privat")
                        with col_comp2:
                            st.metric("Siap Kerja", f"{privat_siap:.1f}%", delta=f"{privat_siap - publik_siap:+.1f}%")
                        with col_comp3:
                            st.metric("Belum Siap", f"{privat_belum:.1f}%", delta=f"{privat_belum - publik_belum:+.1f}%")
                        
                        # Grafik Distribusi Target
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        colors = ['#4ECDC4', '#FF6B6B']
                        target_counts = df_batch['Prediksi'].value_counts()
                        sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=colors)
                        axes[0].set_title('Distribusi Hasil Prediksi - Data Privat')
                        axes[0].set_xlabel('Status')
                        axes[0].set_ylabel('Jumlah')
                        for i, v in enumerate(target_counts.values):
                            axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')
                        axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                        axes[1].set_title('Proporsi Hasil Prediksi - Data Privat')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # D. DISTRIBUSI PROBABILITAS PREDIKSI
                        st.markdown('<p class="section-title">D. Distribusi Probabilitas Prediksi</p>', unsafe_allow_html=True)
                        
                        prob_mean = np.mean(probabilities)
                        prob_std = np.std(probabilities)
                        prob_min = np.min(probabilities)
                        prob_max = np.max(probabilities)
                        
                        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                        with col_d1:
                            st.metric("Rata-rata", f"{prob_mean:.3f}")
                        with col_d2:
                            st.metric("Std Deviasi", f"{prob_std:.3f}")
                        with col_d3:
                            st.metric("Minimum", f"{prob_min:.3f}")
                        with col_d4:
                            st.metric("Maksimum", f"{prob_max:.3f}")
                        
                        st.markdown("**Interpretasi Tingkat Keyakinan Model:**")
                        if prob_mean > 0.7:
                            st.success(f"Model sangat yakin (rata-rata probabilitas {prob_mean*100:.1f}%)")
                        elif prob_mean > 0.6:
                            st.info(f"Model cukup yakin (rata-rata probabilitas {prob_mean*100:.1f}%)")
                        else:
                            st.warning(f"Model kurang yakin (rata-rata probabilitas {prob_mean*100:.1f}%)")
                        
                        # Histogram probabilitas
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(probabilities, bins=20, color='#667eea', edgecolor='black', alpha=0.7)
                        ax.axvline(prob_mean, color='red', linestyle='--', label=f'Rata-rata: {prob_mean:.3f}')
                        ax.axvline(0.5, color='orange', linestyle='--', label='Threshold: 0.5')
                        ax.set_xlabel('Probabilitas')
                        ax.set_ylabel('Jumlah')
                        ax.set_title('Distribusi Probabilitas Prediksi')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                        
                        # E. PERBANDINGAN POLA FAKTOR DOMINAN
                        st.markdown('<p class="section-title">E. Perbandingan Pola Faktor Dominan (Rata-rata per Fitur)</p>', unsafe_allow_html=True)
                        
                        feature_importance_publik = pd.DataFrame({
                            'feature': all_attrs,
                            'importance': model.feature_importances_
                        })
                        
                        mean_attr_privat = df_batch[all_attrs].mean()
                        feature_importance_privat = pd.DataFrame({
                            'feature': all_attrs,
                            'importance': mean_attr_privat / mean_attr_privat.sum()
                        })
                        
                        soft_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(soft_attrs)]['importance'].mean()
                        life_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(life_attrs)]['importance'].mean()
                        tech_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(tech_attrs)]['importance'].mean()
                        total_avg_publik = soft_avg_publik + life_avg_publik + tech_avg_publik
                        
                        soft_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(soft_attrs)]['importance'].mean()
                        life_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(life_attrs)]['importance'].mean()
                        tech_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(tech_attrs)]['importance'].mean()
                        total_avg_privat = soft_avg_privat + life_avg_privat + tech_avg_privat
                        
                        top5_publik = feature_importance_publik.nlargest(5, 'importance')
                        st.markdown("**Top 5 Fitur Dominan - Dataset Publik:**")
                        for i, row in top5_publik.iterrows():
                            label = row['feature'].replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').replace('ratelife', 'Rate Life').replace('ratetech', 'Rate Tech').title()
                            st.write(f"  {label}: {row['importance']:.4f}")
                        
                        top5_privat = feature_importance_privat.nlargest(5, 'importance')
                        st.markdown("**Top 5 Fitur Dominan - Dataset Privat:**")
                        for i, row in top5_privat.iterrows():
                            label = row['feature'].replace('-', ' ').replace('                                                         ', '').replace('skill(CONFIDENCE)', '').replace('ratesoft', 'Rate Soft').replace('ratelife', 'Rate Life').replace('ratetech', 'Rate Tech').title()
                            st.write(f"  {label}: {row['importance']:.4f}")
                        
                        st.markdown("**Kontribusi per Kelompok (Rata-rata per Fitur):**")
                        
                        comp_df = pd.DataFrame({
                            'Kelompok': ['Soft Skills', 'Life Skills', 'Technical Skills'],
                            'Jumlah Fitur': [len(soft_attrs), len(life_attrs), len(tech_attrs)],
                            'Publik (%)': [
                                soft_avg_publik / total_avg_publik * 100,
                                life_avg_publik / total_avg_publik * 100,
                                tech_avg_publik / total_avg_publik * 100
                            ],
                            'Privat (%)': [
                                soft_avg_privat / total_avg_privat * 100,
                                life_avg_privat / total_avg_privat * 100,
                                tech_avg_privat / total_avg_privat * 100
                            ]
                        })
                        
                        st.dataframe(comp_df.round(2), use_container_width=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        x = np.arange(len(comp_df['Kelompok']))
                        width = 0.35
                        
                        ax.bar(x - width/2, comp_df['Publik (%)'], width, label='Publik', color='#667eea')
                        ax.bar(x + width/2, comp_df['Privat (%)'], width, label='Privat', color='#f093fb')
                        ax.set_xlabel('Kelompok')
                        ax.set_ylabel('Kontribusi (%)')
                        ax.set_title('Perbandingan Kontribusi per Kelompok (Rata-rata per Fitur)')
                        ax.set_xticks(x)
                        ax.set_xticklabels(comp_df['Kelompok'])
                        ax.legend()
                        
                        for i, (pub, pri) in enumerate(zip(comp_df['Publik (%)'], comp_df['Privat (%)'])):
                            ax.text(i - width/2, pub + 0.5, f'{pub:.1f}%', ha='center', fontsize=9)
                            ax.text(i + width/2, pri + 0.5, f'{pri:.1f}%', ha='center', fontsize=9)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        st.markdown("**Interpretasi Perbedaan Pola Faktor Dominan:**")
                        
                        if soft_avg_publik > life_avg_publik and soft_avg_publik > tech_avg_publik:
                            dominan_publik = "Soft Skills"
                        elif life_avg_publik > soft_avg_publik and life_avg_publik > tech_avg_publik:
                            dominan_publik = "Life Skills"
                        else:
                            dominan_publik = "Technical Skills"
                        
                        if soft_avg_privat > life_avg_privat and soft_avg_privat > tech_avg_privat:
                            dominan_privat = "Soft Skills"
                        elif life_avg_privat > soft_avg_privat and life_avg_privat > tech_avg_privat:
                            dominan_privat = "Life Skills"
                        else:
                            dominan_privat = "Technical Skills"
                        
                        st.write(f"  - **Dataset Publik:** Faktor dominan adalah **{dominan_publik}**")
                        st.write(f"  - **Dataset Privat:** Faktor dominan adalah **{dominan_privat}**")
                        
                        if dominan_publik == dominan_privat:
                            st.success(f"  Pola dominan konsisten: {dominan_publik} mendominasi di kedua dataset")
                        else:
                            st.warning(f"  Pola dominan berbeda: Publik ({dominan_publik}) vs Privat ({dominan_privat})")
                        
                        # F. ANALISIS STATISTIK DESKRIPTIF
                        st.markdown('<p class="section-title">F. Analisis Statistik Deskriptif</p>', unsafe_allow_html=True)
                        
                        soft_data = df_batch[soft_attrs].values.flatten()
                        life_data = df_batch[life_attrs].values.flatten()
                        tech_data = df_batch[tech_attrs].values.flatten()
                        
                        stats_data = {
                            'Kelompok': ['Soft Skills', 'Life Skills', 'Technical Skills'],
                            'Mean': [np.mean(soft_data), np.mean(life_data), np.mean(tech_data)],
                            'Std Dev': [np.std(soft_data), np.std(life_data), np.std(tech_data)],
                            'Min': [np.min(soft_data), np.min(life_data), np.min(tech_data)],
                            'Max': [np.max(soft_data), np.max(life_data), np.max(tech_data)]
                        }
                        stats_df = pd.DataFrame(stats_data)
                        
                        st.dataframe(stats_df.round(3), use_container_width=True)
                        
                        st.markdown("**Interpretasi Hasil Statistik Deskriptif:**")
                        for _, row in stats_df.iterrows():
                            st.write(f"  - {row['Kelompok']}: rata-rata {row['Mean']:.2f} (std {row['Std Dev']:.2f}, min {row['Min']:.0f}, max {row['Max']:.0f})")
                        
                        st.markdown('<p class="section-title">Detail Hasil Prediksi</p>', unsafe_allow_html=True)
                        st.dataframe(df_batch[['Nama_Mahasiswa'] + all_attrs + ['Prediksi', 'Probabilitas']], use_container_width=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_batch.to_excel(writer, sheet_name='Hasil Prediksi', index=False)
                        st.download_button(
                            label="Download Hasil Prediksi",
                            data=output.getvalue(),
                            file_name="hasil_prediksi.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"Error membaca file: {e}")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    Prediksi Kesiapan Kerja Mahasiswa | Random Forest Model | Akurasi 67.08%
</div>
""", unsafe_allow_html=True)