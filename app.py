import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Kesiapan Kerja Mahasiswa",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CSS CUSTOM
# ============================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: white;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box.siap {
        background: linear-gradient(135deg, #d4edda, #b7e4c7);
        border: 2px solid #28a745;
    }
    .prediction-box.belum {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .prediction-box h2 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .prediction-box .probability {
        font-size: 1.2rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .prediction-box .student-info {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
    }
    .recommendation-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        font-size: 0.95rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    .recommendation-box.improve {
        background: #dc3545;
        border-left: 6px solid #a71d2a;
    }
    .recommendation-box.maintain {
        background: #198754;
        border-left: 6px solid #0f5132;
    }
    .recommendation-box .attribute {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .recommendation-box .suggestion {
        font-size: 0.88rem;
        font-weight: 400;
        line-height: 1.5;
        opacity: 0.98;
    }
    .section-title {
        color: #1a1a2e;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .student-info-card {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #888;
        font-size: 0.85rem;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
    }
    .stDownloadButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(40,167,69,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>Prediksi Kesiapan Kerja Mahasiswa</h1>
    <p>Berbasis Soft Skills, Life Skills, dan Technical Skills</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (TANPA SCALER)
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    return model


try:
    model = load_model()
    all_attrs = list(model.feature_names_in_)   # 36 fitur dengan prefix SS_/LS_/TS_
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file 'random_forest_model.pkl' berada di folder yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat load model: {e}")
    st.stop()


# ============================================================
# BAGI FITUR PER KELOMPOK (PAKAI PREFIX)
# ============================================================

soft_attrs = [c for c in all_attrs if c.startswith('SS_')]
life_attrs = [c for c in all_attrs if c.startswith('LS_')]
tech_attrs = [c for c in all_attrs if c.startswith('TS_')]


# ============================================================
# FUNGSI FORMAT NAMA ATRIBUT
# ============================================================

def format_attribute_name(attr):
    """Hilangkan prefix SS_/LS_/TS_ dan rapikan nama."""
    label = attr
    for prefix in ('SS_', 'LS_', 'TS_'):
        if attr.startswith(prefix):
            label = attr[len(prefix):]
            break
    return label.replace('-', ' ').title()


# ============================================================
# FUNGSI SARAN PENGEMBANGAN
# ============================================================

def get_suggestion(attr):
    attr_lower = attr.lower()
    suggestions = {
        'decision': "Tingkatkan kemampuan pengambilan keputusan melalui latihan studi kasus, simulasi pengambilan keputusan, dan evaluasi terhadap keputusan yang telah dibuat.",
        'planning': "Latih kemampuan perencanaan dengan membuat target harian dan mingguan, menyusun prioritas pekerjaan, serta membuat jadwal penyelesaian tugas.",
        'teamwork': "Tingkatkan pengalaman kerja sama dengan lebih aktif dalam proyek kelompok, diskusi, organisasi, dan pembagian tugas.",
        'confident': "Tingkatkan kepercayaan diri melalui latihan presentasi, berbicara di depan kelompok, dan menyampaikan pendapat secara aktif.",
        'meet': "Latih kemampuan memenuhi tenggat waktu dengan membuat jadwal kerja, menentukan prioritas, dan menyelesaikan tugas sebelum batas waktu.",
        'stress': "Tingkatkan kemampuan menghadapi situasi penuh tekanan melalui latihan pemecahan masalah, pengaturan waktu, dan simulasi kondisi kerja.",
        'communicate': "Tingkatkan kemampuan komunikasi dengan membiasakan diri menyampaikan informasi secara jelas, mendengarkan orang lain, dan melakukan diskusi secara aktif.",
        'boosting': "Kembangkan kreativitas dengan mencoba metode baru, melakukan brainstorming, mengikuti proyek kreatif, dan mencari berbagai alternatif solusi.",
        'multitask': "Latih kemampuan mengelola beberapa pekerjaan dengan menentukan prioritas, membuat daftar tugas, dan mengalokasikan waktu untuk setiap pekerjaan.",
        'working': "Tingkatkan kemampuan bekerja secara mandiri dengan membiasakan menyelesaikan tugas tanpa selalu bergantung pada bantuan orang lain.",
        'opinion': "Latih keberanian menyampaikan pendapat melalui diskusi, presentasi, dan forum kelompok dengan menggunakan argumen yang logis.",
        'resolve': "Tingkatkan kemampuan menyelesaikan konflik dengan belajar mendengarkan berbagai sudut pandang, mencari akar masalah, dan menentukan solusi bersama.",
        'convey': "Latih kemampuan menyampaikan informasi dengan struktur yang jelas, bahasa yang mudah dipahami, serta contoh yang relevan.",
        'alone': "Tingkatkan kemandirian melalui latihan menyelesaikan tugas secara mandiri dan mengambil tanggung jawab terhadap hasil pekerjaan.",
        'rearranging': "Latih kemampuan beradaptasi dengan membiasakan diri menyesuaikan rencana ketika terjadi perubahan kondisi atau prioritas pekerjaan.",
        'inspiration': "Perbanyak referensi dan pengalaman baru melalui membaca, mengikuti seminar, mengamati lingkungan sekitar, dan mencoba berbagai aktivitas baru.",
        'motivation': "Tingkatkan motivasi dengan menetapkan tujuan karier, membuat target yang terukur, dan mengevaluasi perkembangan secara berkala.",
        'emotional': "Latih kemampuan mengelola emosi dengan mengenali kondisi diri, mengendalikan respons terhadap tekanan, dan menjaga komunikasi tetap positif.",
        'demonstrator': "Tingkatkan kemampuan demonstrasi dengan lebih sering mempraktikkan keterampilan dan menjelaskan prosesnya kepada orang lain.",
        'good leader': "Kembangkan kepemimpinan dengan mengambil tanggung jawab dalam kelompok, membagi tugas secara adil, dan membantu anggota mencapai tujuan bersama.",
        'good listener': "Latih kemampuan mendengarkan dengan memberikan perhatian penuh, tidak memotong pembicaraan, dan memberikan respons yang sesuai.",
        'oral': "Tingkatkan kemampuan komunikasi lisan melalui presentasi, diskusi, wawancara simulasi, dan latihan berbicara.",
        'team building': "Perbanyak kegiatan team building melalui proyek kelompok, organisasi, kegiatan kelas, dan aktivitas kolaboratif.",
        'area of interest': "Perjelas bidang minat dengan mengeksplorasi berbagai bidang pekerjaan, mengikuti pelatihan, dan mencoba proyek yang sesuai dengan minat.",
        'good presenter': "Tingkatkan kemampuan presentasi dengan latihan rutin, memperbaiki struktur materi, dan meningkatkan kemampuan berkomunikasi di depan audiens.",
        'coach': "Tingkatkan kemampuan membimbing orang lain dengan belajar memberikan arahan, umpan balik, dan dukungan secara efektif.",
        'interpersonal': "Tingkatkan kemampuan interpersonal melalui komunikasi positif, kerja sama, empati, dan membangun hubungan yang baik dengan orang lain.",
        'time management': "Tingkatkan manajemen waktu dengan membuat jadwal, menentukan prioritas, dan menggunakan target waktu untuk setiap pekerjaan.",
        'quality management': "Tingkatkan kemampuan menjaga kualitas dengan membuat standar pekerjaan, melakukan pengecekan ulang, dan mengevaluasi hasil pekerjaan.",
        'blend': "Perbanyak latihan menggabungkan berbagai keterampilan melalui proyek yang mengintegrasikan kemampuan teknis dan nonteknis.",
        'gain': "Tingkatkan kemampuan melalui pelatihan, kursus, praktik langsung, dan evaluasi hasil belajar secara berkala.",
        'new hands-on': "Perbanyak pengalaman praktik langsung melalui proyek, praktikum, magang, atau kegiatan yang berhubungan dengan dunia kerja.",
        'group project': "Tingkatkan pengalaman proyek kelompok dengan mengambil peran aktif, menyelesaikan tanggung jawab, dan berkoordinasi dengan anggota kelompok.",
        'developing projects': "Latih kemampuan pengembangan proyek dengan membuat proyek kecil secara bertahap mulai dari perencanaan hingga evaluasi.",
        'cost/time': "Tingkatkan kemampuan mengelola biaya dan waktu dengan membuat estimasi, menentukan prioritas, dan melakukan evaluasi penggunaan sumber daya.",
        'administrative': "Tingkatkan kemampuan administratif melalui latihan pengelolaan dokumen, pencatatan data, penyusunan laporan, dan pengarsipan."
    }
    for keyword, suggestion in suggestions.items():
        if keyword in attr_lower:
            return suggestion
    return "Tingkatkan atribut ini melalui latihan rutin, pengalaman praktik, kegiatan kelompok, pelatihan, serta evaluasi perkembangan secara berkala."


# ============================================================
# TEMPLATE EXCEL
# ============================================================

def create_template():
    df_template = pd.DataFrame(columns=all_attrs + ['Nama_Mahasiswa', 'NIM', 'Umur', 'Jurusan'])
    df_template.loc[0] = [2] * len(all_attrs) + ['Contoh Mahasiswa', '123456789', '21', 'Teknik Informatika']
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, sheet_name='Template', index=False)
    return output.getvalue()


# ============================================================
# FUNGSI PREDIKSI (TANPA SCALER)
# ============================================================

def predict_single(input_data):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[all_attrs]
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    return prediction, probability


def predict_batch(df):
    df = df[all_attrs]
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
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


# ============================================================
# TAB INPUT
# ============================================================

tab1, tab2 = st.tabs(["📝 Input Manual", "📊 Input Excel"])


# ============================================================
# TAB 1 - INPUT MANUAL
# ============================================================

with tab1:
    st.markdown('<p class="section-title">📝 Input Data Mahasiswa</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown("### 👤 Data Diri Mahasiswa")
        col_n1, col_n2, col_n3, col_n4 = st.columns(4)

        with col_n1:
            nama_mahasiswa = st.text_input("Nama Mahasiswa", placeholder="Masukkan nama lengkap", key="nama_input")
        with col_n2:
            nim_mahasiswa = st.text_input("NIM", placeholder="Masukkan NIM", key="nim_input")
        with col_n3:
            umur_mahasiswa = st.number_input("Umur", min_value=17, max_value=60, value=20, step=1, key="umur_input")
        with col_n4:
            jurusan_mahasiswa = st.selectbox(
                "Jurusan",
                options=[
                    "Teknik Informatika", "Sistem Informasi", "Teknik Elektro",
                    "Teknik Mesin", "Teknik Sipil", "Manajemen", "Akuntansi",
                    "Ekonomi", "Hukum", "Psikologi", "Lainnya"
                ],
                key="jurusan_input"
            )

    st.markdown("---")

    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        st.subheader(f"🧠 Soft Skills ({len(soft_attrs)} atribut)")
        for attr in soft_attrs:
            label = format_attribute_name(attr)
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"soft_{attr}")

    with col2:
        st.subheader(f"💪 Life Skills ({len(life_attrs)} atribut)")
        for attr in life_attrs:
            label = format_attribute_name(attr)
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"life_{attr}")

        st.subheader(f"🛠️ Technical Skills ({len(tech_attrs)} atribut)")
        for attr in tech_attrs:
            label = format_attribute_name(attr)
            input_data[attr] = st.slider(label, 0, 5, 2, key=f"tech_{attr}")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_manual = st.button("🔮 Prediksi Kesiapan Kerja", type="primary", use_container_width=True)

    if predict_manual:
        # ========================================
        # VALIDASI NAMA & NIM (SESUAI FLOWCHART)
        # ========================================
        if not nama_mahasiswa.strip():
            st.warning("⚠️ Silakan masukkan Nama Mahasiswa terlebih dahulu!")
            st.stop()
        if not nim_mahasiswa.strip():
            st.warning("⚠️ Silakan masukkan NIM terlebih dahulu!")
            st.stop()

        with st.spinner("Sistem menganalisis data..."):
            prediction, probability = predict_single(input_data)
            contributions = get_feature_importance()
            soft_score = np.mean([input_data[attr] for attr in soft_attrs])
            life_score = np.mean([input_data[attr] for attr in life_attrs])
            tech_score = np.mean([input_data[attr] for attr in tech_attrs])

        st.markdown("---")
        st.markdown("### 👤 Informasi Mahasiswa")
        st.markdown(f"""
        <div class="student-info-card">
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 4px 8px; font-weight: 600; color: #555; width: 15%;">Nama</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #1a1a2e; width: 35%;">{nama_mahasiswa}</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #555; width: 15%;">NIM</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #1a1a2e; width: 35%;">{nim_mahasiswa}</td>
                </tr>
                <tr>
                    <td style="padding: 4px 8px; font-weight: 600; color: #555;">Umur</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #1a1a2e;">{umur_mahasiswa} tahun</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #555;">Jurusan</td>
                    <td style="padding: 4px 8px; font-weight: 600; color: #1a1a2e;">{jurusan_mahasiswa}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ========================================
        # SISTEM MENAMPILKAN HASIL (SESUAI FLOWCHART)
        # ========================================
        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-box siap">
                    <h2 style="color:#155724;">✅ SIAP KERJA</h2>
                    <div class="probability">Probabilitas: {probability * 100:.1f}%</div>
                    <div class="student-info">📌 {nama_mahasiswa} | {nim_mahasiswa} | {jurusan_mahasiswa}</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box belum">
                    <h2 style="color:#721c24;">❌ BELUM SIAP KERJA</h2>
                    <div class="probability">Probabilitas: {probability * 100:.1f}%</div>
                    <div class="student-info">📌 {nama_mahasiswa} | {nim_mahasiswa} | {jurusan_mahasiswa}</div>
                </div>
                """, unsafe_allow_html=True
            )

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{soft_score:.2f}</div>
                    <div class="label">🧠 Soft Skills</div>
                </div>
                """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{life_score:.2f}</div>
                    <div class="label">💪 Life Skills</div>
                </div>
                """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{tech_score:.2f}</div>
                    <div class="label">🛠️ Technical Skills</div>
                </div>
                """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown('<p class="section-title">📊 Radar Chart Skor per Kelompok</p>', unsafe_allow_html=True)
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
            st.markdown('<p class="section-title">📊 Kontribusi Faktor</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            categories = list(contributions.keys())
            values = list(contributions.values())
            colors = ['#667eea', '#764ba2', '#f093fb']
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Kontribusi (%)')
            ax.set_title('Feature Importance per Kelompok', fontsize=12)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        st.markdown('<p class="section-title">📋 Analisis dan Rekomendasi Pengembangan</p>', unsafe_allow_html=True)
        st.info("🔴 Skor 0–2 = Perlu Ditingkatkan  |  🟡 Skor 3 = Cukup  |  🟢 Skor 4–5 = Perlu Dipertahankan")

        all_scores = {attr: input_data[attr] for attr in all_attrs}
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1])
        low_attrs = [(attr, score) for attr, score in sorted_scores if score <= 2]
        high_attrs = [(attr, score) for attr, score in sorted_scores if score >= 4]

        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            st.markdown("### 🔴 Perlu Ditingkatkan")
            st.caption("Atribut dengan skor ≤ 2 perlu mendapatkan perhatian dan pengembangan lebih lanjut.")
            if low_attrs:
                for attr, score in low_attrs[:10]:
                    label = format_attribute_name(attr)
                    suggestion = get_suggestion(attr)
                    st.markdown(f"""
                        <div class="recommendation-box improve">
                            <div class="attribute">🔴 {label} — Skor {score}</div>
                            <div class="suggestion"><b>Saran:</b> {suggestion}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ Tidak terdapat atribut dengan skor ≤ 2. Kemampuan mahasiswa berada pada kondisi baik.")

        with col_rec2:
            st.markdown("### 🟢 Perlu Dipertahankan")
            st.caption("Atribut dengan skor ≥ 4 menunjukkan kemampuan yang sudah baik dan perlu dipertahankan.")
            if high_attrs:
                for attr, score in high_attrs[:10]:
                    label = format_attribute_name(attr)
                    st.markdown(f"""
                        <div class="recommendation-box maintain">
                            <div class="attribute">🟢 {label} — Skor {score}</div>
                            <div class="suggestion"><b>Saran:</b> Pertahankan kemampuan ini dengan terus menerapkannya dalam perkuliahan, proyek, organisasi, magang, dan aktivitas kerja.</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Belum terdapat atribut dengan skor ≥ 4. Fokus utama adalah meningkatkan atribut dengan skor rendah.")

        if low_attrs:
            st.markdown("### 🎯 Prioritas Pengembangan")
            priority_attr = low_attrs[0]
            priority_name = format_attribute_name(priority_attr[0])
            priority_score = priority_attr[1]
            priority_suggestion = get_suggestion(priority_attr[0])
            st.warning(f"**{nama_mahasiswa}** - Prioritas utama adalah **{priority_name}** dengan skor **{priority_score}**. {priority_suggestion}")

        st.markdown('<p class="section-title">📊 Detail Skor per Atribut</p>', unsafe_allow_html=True)
        tab_soft, tab_life, tab_tech = st.tabs(["🧠 Soft Skills", "💪 Life Skills", "🛠️ Technical Skills"])

        with tab_soft:
            df_soft = pd.DataFrame({
                'Atribut': [format_attribute_name(a) for a in soft_attrs],
                'Skor': [input_data[attr] for attr in soft_attrs]
            })
            df_soft['Skor'] = df_soft['Skor'].astype(int)
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#28a745' if s >= 4 else '#ffc107' if s == 3 else '#dc3545' for s in df_soft['Skor']]
            bars = ax.barh(df_soft['Atribut'], df_soft['Skor'], color=colors)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Skor')
            ax.set_title('Soft Skills')
            for bar, val in zip(bars, df_soft['Skor']):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, str(val), va='center')
            st.pyplot(fig)
            plt.close()

        with tab_life:
            df_life = pd.DataFrame({
                'Atribut': [format_attribute_name(a) for a in life_attrs],
                'Skor': [input_data[attr] for attr in life_attrs]
            })
            df_life['Skor'] = df_life['Skor'].astype(int)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#28a745' if s >= 4 else '#ffc107' if s == 3 else '#dc3545' for s in df_life['Skor']]
            bars = ax.barh(df_life['Atribut'], df_life['Skor'], color=colors)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Skor')
            ax.set_title('Life Skills')
            for bar, val in zip(bars, df_life['Skor']):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, str(val), va='center')
            st.pyplot(fig)
            plt.close()

        with tab_tech:
            df_tech = pd.DataFrame({
                'Atribut': [format_attribute_name(a) for a in tech_attrs],
                'Skor': [input_data[attr] for attr in tech_attrs]
            })
            df_tech['Skor'] = df_tech['Skor'].astype(int)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#28a745' if s >= 4 else '#ffc107' if s == 3 else '#dc3545' for s in df_tech['Skor']]
            bars = ax.barh(df_tech['Atribut'], df_tech['Skor'], color=colors)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Skor')
            ax.set_title('Technical Skills')
            for bar, val in zip(bars, df_tech['Skor']):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, str(val), va='center')
            st.pyplot(fig)
            plt.close()

        # ========================================
        # DOWNLOAD HASIL PREDIKSI (SESUAI FLOWCHART 1)
        # ========================================
        st.markdown('<p class="section-title">📥 Download Hasil Prediksi</p>', unsafe_allow_html=True)
        st.caption("Hasil prediksi untuk mahasiswa ini dapat diunduh dalam format Excel.")

        df_hasil_manual = pd.DataFrame({
            'Nama_Mahasiswa': [nama_mahasiswa],
            'NIM': [nim_mahasiswa],
            'Umur': [umur_mahasiswa],
            'Jurusan': [jurusan_mahasiswa],
            **{attr: [input_data[attr]] for attr in all_attrs},
            'Prediksi': ['Siap Kerja' if prediction == 1 else 'Belum Siap'],
            'Probabilitas': [round(float(probability), 4)]
        })

        output_manual = BytesIO()
        with pd.ExcelWriter(output_manual, engine='openpyxl') as writer:
            df_hasil_manual.to_excel(writer, sheet_name='Hasil Prediksi', index=False)

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="📥 Download Hasil Prediksi (Excel)",
                data=output_manual.getvalue(),
                file_name=f"hasil_prediksi_{nim_mahasiswa}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_manual_single"
            )


# ============================================================
# TAB 2 - INPUT EXCEL
# ============================================================

with tab2:
    st.markdown('<p class="section-title">📊 Upload Data Excel</p>', unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    with col_t2:
        template_data = create_template()
        st.download_button(
            label="📥 Download Template Excel",
            data=template_data,
            file_name="template_prediksi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_template"
        )

    st.markdown("""
    **Petunjuk:**
    1. Download template Excel di atas.
    2. Isi data mahasiswa dengan skala 0–5.
    3. Kolom yang tersedia: **Nama_Mahasiswa**, **NIM**, **Umur**, **Jurusan** + semua atribut skills.
    4. Pastikan nama kolom tidak diubah.
    5. Upload file Excel yang sudah diisi.
    """)

    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=['xlsx'])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_excel(uploaded_file)
            missing_cols = [col for col in all_attrs if col not in df_batch.columns]

            # ========================================
            # SISTEM VALIDASI KOLOM (SESUAI FLOWCHART 2)
            # ========================================
            if missing_cols:
                st.error(f"❌ Kolom tidak sesuai! Kolom berikut tidak ditemukan: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
                st.info("Silakan download ulang template Excel dan pastikan nama kolom tidak diubah.")
            else:
                st.success(f"✅ Kolom sesuai. Data berhasil dimuat: {len(df_batch)} responden")

                if st.button("🔮 Prediksi Semua Data", type="primary", use_container_width=True):
                    with st.spinner("Sistem memproses semua data..."):
                        predictions, probabilities = predict_batch(df_batch)
                        df_batch['Prediksi'] = ['Siap Kerja' if p == 1 else 'Belum Siap' for p in predictions]
                        df_batch['Probabilitas'] = probabilities

                    st.markdown('<p class="section-title">📊 Hasil Prediksi Kesiapan Kerja</p>', unsafe_allow_html=True)

                    total = len(df_batch)
                    siap = df_batch[df_batch['Prediksi'] == 'Siap Kerja'].shape[0]
                    belum = df_batch[df_batch['Prediksi'] == 'Belum Siap'].shape[0]

                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        st.metric("Total Responden", total)
                    with col_c2:
                        st.metric("✅ Siap Kerja", f"{siap} ({siap / total * 100:.1f}%)")
                    with col_c3:
                        st.metric("❌ Belum Siap", f"{belum} ({belum / total * 100:.1f}%)")

                    st.markdown('<p class="section-title">📈 Perbandingan dengan Dataset Publik (Mendeley)</p>', unsafe_allow_html=True)

                    publik_siap = 28.8
                    publik_belum = 71.2
                    privat_siap = siap / total * 100
                    privat_belum = belum / total * 100

                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    with col_comp1:
                        st.metric("Dataset", "Privat")
                    with col_comp2:
                        st.metric("Siap Kerja", f"{privat_siap:.1f}%", delta=f"{privat_siap - publik_siap:+.1f}%")
                    with col_comp3:
                        st.metric("Belum Siap", f"{privat_belum:.1f}%", delta=f"{privat_belum - publik_belum:+.1f}%")

                    st.markdown('<p class="section-title">📊 Distribusi Hasil Prediksi</p>', unsafe_allow_html=True)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    colors = ['#28a745', '#dc3545']
                    target_counts = df_batch['Prediksi'].value_counts()

                    sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=colors)
                    axes[0].set_title('Distribusi Hasil Prediksi - Data Privat')
                    axes[0].set_xlabel('Status')
                    axes[0].set_ylabel('Jumlah')
                    for i, v in enumerate(target_counts.values):
                        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

                    axes[1].pie(target_counts.values, labels=target_counts.index,
                                autopct='%1.1f%%', colors=colors, startangle=90)
                    axes[1].set_title('Proporsi Hasil Prediksi - Data Privat')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown('<p class="section-title">📊 Distribusi Probabilitas Prediksi</p>', unsafe_allow_html=True)

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
                        st.success(f"Model sangat yakin (rata-rata probabilitas {prob_mean * 100:.1f}%)")
                    elif prob_mean > 0.6:
                        st.info(f"Model cukup yakin (rata-rata probabilitas {prob_mean * 100:.1f}%)")
                    else:
                        st.warning(f"Model kurang yakin (rata-rata probabilitas {prob_mean * 100:.1f}%)")

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

                    st.markdown('<p class="section-title">📊 Perbandingan Pola Faktor Dominan (Rata-rata per Fitur)</p>', unsafe_allow_html=True)

                    feature_importance_publik = pd.DataFrame({
                        'feature': all_attrs,
                        'importance': model.feature_importances_
                    })

                    mean_attr_privat = df_batch[all_attrs].mean()
                    feature_importance_privat = pd.DataFrame({
                        'feature': all_attrs,
                        'importance': mean_attr_privat / mean_attr_privat.sum()
                    })

                    top5_publik = feature_importance_publik.nlargest(5, 'importance')
                    st.markdown("**Top 5 Fitur Dominan - Dataset Publik:**")
                    for _, row in top5_publik.iterrows():
                        label = format_attribute_name(row['feature'])
                        st.write(f"• {label}: {row['importance']:.4f}")

                    top5_privat = feature_importance_privat.nlargest(5, 'importance')
                    st.markdown("**Top 5 Fitur Dominan - Dataset Privat:**")
                    for _, row in top5_privat.iterrows():
                        label = format_attribute_name(row['feature'])
                        st.write(f"• {label}: {row['importance']:.4f}")

                    st.markdown("**Kontribusi per Kelompok (Rata-rata per Fitur):**")

                    soft_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(soft_attrs)]['importance'].mean()
                    life_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(life_attrs)]['importance'].mean()
                    tech_avg_publik = feature_importance_publik[feature_importance_publik['feature'].isin(tech_attrs)]['importance'].mean()
                    total_avg_publik = soft_avg_publik + life_avg_publik + tech_avg_publik

                    soft_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(soft_attrs)]['importance'].mean()
                    life_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(life_attrs)]['importance'].mean()
                    tech_avg_privat = feature_importance_privat[feature_importance_privat['feature'].isin(tech_attrs)]['importance'].mean()
                    total_avg_privat = soft_avg_privat + life_avg_privat + tech_avg_privat

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

                    ax.bar(x - width / 2, comp_df['Publik (%)'], width, label='Publik', color='#667eea')
                    ax.bar(x + width / 2, comp_df['Privat (%)'], width, label='Privat', color='#f093fb')
                    ax.set_xlabel('Kelompok')
                    ax.set_ylabel('Kontribusi (%)')
                    ax.set_title('Perbandingan Kontribusi per Kelompok (Rata-rata per Fitur)')
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df['Kelompok'])
                    ax.legend()

                    for i, (pub, pri) in enumerate(zip(comp_df['Publik (%)'], comp_df['Privat (%)'])):
                        ax.text(i - width / 2, pub + 0.5, f'{pub:.1f}%', ha='center', fontsize=9)
                        ax.text(i + width / 2, pri + 0.5, f'{pri:.1f}%', ha='center', fontsize=9)

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

                    st.write(f"• **Dataset Publik:** Faktor dominan adalah **{dominan_publik}**")
                    st.write(f"• **Dataset Privat:** Faktor dominan adalah **{dominan_privat}**")

                    if dominan_publik == dominan_privat:
                        st.success(f"✅ Pola dominan konsisten: {dominan_publik} mendominasi di kedua dataset.")
                    else:
                        st.warning(f"⚠️ Pola dominan berbeda: Publik ({dominan_publik}) vs Privat ({dominan_privat}).")

                    st.markdown('<p class="section-title">📊 Analisis Statistik Deskriptif</p>', unsafe_allow_html=True)

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
                        st.write(f"• {row['Kelompok']}: rata-rata {row['Mean']:.2f} "
                                 f"(std {row['Std Dev']:.2f}, min {row['Min']:.0f}, max {row['Max']:.0f})")

                    st.markdown('<p class="section-title">📋 Detail Hasil Prediksi</p>', unsafe_allow_html=True)

                    columns_to_show = []
                    if 'Nama_Mahasiswa' in df_batch.columns:
                        columns_to_show.append('Nama_Mahasiswa')
                    if 'NIM' in df_batch.columns:
                        columns_to_show.append('NIM')
                    if 'Umur' in df_batch.columns:
                        columns_to_show.append('Umur')
                    if 'Jurusan' in df_batch.columns:
                        columns_to_show.append('Jurusan')
                    columns_to_show += all_attrs + ['Prediksi', 'Probabilitas']

                    st.dataframe(df_batch[columns_to_show], use_container_width=True)

                    # ========================================
                    # DOWNLOAD HASIL PREDIKSI BATCH (SESUAI FLOWCHART 2)
                    # ========================================
                    st.markdown('<p class="section-title">📥 Download Hasil Prediksi Excel</p>', unsafe_allow_html=True)

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_batch.to_excel(writer, sheet_name='Hasil Prediksi', index=False)

                    col_dl_b1, col_dl_b2, col_dl_b3 = st.columns([1, 2, 1])
                    with col_dl_b2:
                        st.download_button(
                            label="📥 Download Hasil Prediksi (Excel)",
                            data=output.getvalue(),
                            file_name="hasil_prediksi.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_batch"
                        )

        except Exception as e:
            st.error(f"Error membaca file: {e}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Prediksi Kesiapan Kerja Mahasiswa | Random Forest Model | Akurasi 64.72%
</div>
""", unsafe_allow_html=True)