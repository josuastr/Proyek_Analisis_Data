import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Seaborn
sns.set(style="whitegrid")

# Fungsi: Load dan Persiapan Data
@st.cache_data
def load_data():
    days_df = pd.read_csv("dashboard/day_cleaned.csv")
    hours_df = pd.read_csv("dashboard/hour_cleaned.csv")
    days_df["tanggal"] = pd.to_datetime(days_df["tanggal"])
    hours_df["tanggal"] = pd.to_datetime(hours_df["tanggal"])
    return days_df, hours_df

# Fungsi: Filter berdasarkan tanggal
def filter_by_date(df, start_date, end_date):
    return df[(df["tanggal"] >= pd.Timestamp(start_date)) & (df["tanggal"] <= pd.Timestamp(end_date))]

# Fungsi: Sidebar dan Filter Tanggal
def sidebar_filters(days_df):
    st.sidebar.title("Proyek Analisis Data: Bike Sharing Dataset")
    st.sidebar.markdown("""
    **Nama:** Josua Sianturi  
    **Email:** [mc281d5y1293@student.devacademy.id](mailto:mc281d5y1293@student.devacademy.id)  
    **ID Dicoding:** MC281D5Y1293  
    """)
    st.sidebar.image("https://miro.medium.com/v2/resize:fit:2000/0*TZ0bsPAR7gGvOoEu", use_column_width=True)
    
    st.sidebar.header("Filter Rentang Waktu")
    st.sidebar.caption("Pilih rentang waktu untuk melihat data penyewaan sepeda.")

    min_date = days_df["tanggal"].min()
    max_date = days_df["tanggal"].max()

    date_range = st.sidebar.date_input(
        "Pilih Rentang Waktu",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    try:
        if len(date_range) == 2:
            return date_range[0], date_range[1]
        else:
            st.warning("⚠️ Silakan pilih **dua tanggal lengkap**.")
            return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memilih tanggal: {e}")
        return None, None
    
# Fungsi: Tampilkan Metrik Total
def show_total_metrics(filtered_days_df):
    if filtered_days_df.empty:
        st.warning("Data kosong. Tidak bisa menampilkan metrik total.")
        return

    total_penyewa_sepeda = filtered_days_df["total_penyewaan_sepeda"].sum()
    total_kasual = filtered_days_df["penyewa_kasual"].sum()
    total_terdaftar = filtered_days_df["penyewa_terdaftar"].sum()

    st.markdown("""
        <style>
        .metric-container {
            position: -webkit-sticky; /* Untuk Safari */
            position: sticky;
            top: 0;
            background-color: #181818;
            z-index: 10;
            padding: 15px;
            border-radius: 10px;
            display: flex;
            justify-content: space-around;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .metric-box {
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box">
                <div>Total Penyewa Sepeda</div>
                <div class="metric-value">{total_penyewa_sepeda:,}</div>
            </div>
            <div class="metric-box">
                <div>Total Penyewa Kasual</div>
                <div class="metric-value">{total_kasual:,}</div>
            </div>
            <div class="metric-box">
                <div>Total Penyewa Terdaftar</div>
                <div class="metric-value">{total_terdaftar:,}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Fungsi: Visualisasi Jam Sibuk/Sepi
def show_hourly_analysis(filtered_hours_df):
    if filtered_hours_df.empty:
        st.warning("Data tidak tersedia untuk jam penyewaan.")
        return

    st.header("Identifikasi Jam Sibuk & Sepi dalam Penyewaan Sepeda")
    hourly_rentals = filtered_hours_df.groupby("jam")["total_penyewaan_sepeda"].sum().reset_index()
    top_busy_hours = hourly_rentals.sort_values(by="total_penyewaan_sepeda", ascending=False).head(5)
    least_busy_hours = hourly_rentals.sort_values(by="total_penyewaan_sepeda", ascending=True).head(5)

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(x="jam", y="total_penyewaan_sepeda", data=top_busy_hours, palette="Blues_r", ax=ax[0])
    ax[0].set_title("Jam Sibuk - Penyewaan Tertinggi")
    sns.barplot(x="jam", y="total_penyewaan_sepeda", data=least_busy_hours, palette="Reds_r", ax=ax[1])
    ax[1].set_title("Jam Sepi - Penyewaan Terendah")
    st.pyplot(fig)

# Fungsi: Clustering Penyewaan Sepeda
def show_clustering(filtered_days_df):
    if filtered_days_df.empty:
        st.warning("Data kosong. Tidak bisa melakukan clustering.")
        return

    st.header("Clustering Penyewaan Sepeda")
    
    st.write("""
    **Apa itu Clustering?**  
    Clustering membantu mengelompokkan data berdasarkan pola yang mirip.  
    Di sini, kita akan mengelompokkan data penyewaan sepeda berdasarkan jumlah penyewa (penyewa_terdaftar atau penyewa_kasual) dan faktor cuaca (suhu, kelembaban, atau kecepatan_angin).
    """)

    feature_options = ["penyewa_terdaftar", "penyewa_kasual", "suhu", "kelembaban", "kecepatan_angin"]
    feature_x = st.selectbox("Pilih fitur untuk sumbu X:", feature_options, index=0)
    feature_y = st.selectbox("Pilih fitur untuk sumbu Y:", feature_options, index=1)

    features = filtered_days_df[[feature_x, feature_y]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    wcss = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 6), wcss, marker="o", linestyle="--", color="b")
    ax.set_title("Metode Elbow untuk Menentukan Jumlah Cluster")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    filtered_days_df["Cluster"] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=filtered_days_df[feature_x],
        y=filtered_days_df[feature_y],
        hue=filtered_days_df["Cluster"],
        palette="Set1",
        s=100,
        ax=ax
    )
    ax.set_title(f"Hasil Clustering Berdasarkan {feature_x} & {feature_y}")
    st.pyplot(fig)

    st.subheader("Interpretasi Hasil Clustering")
    st.write("""
    1.  **Cluster 0**: Penyewaan rendah, kemungkinan terjadi saat suhu dingin atau kelembaban tinggi.  
    2.  **Cluster 1**: Penyewaan tinggi, dominasi oleh penyewa terdaftar. Biasanya terjadi saat suhu optimal.  
    3.  **Cluster 2**: Penyewaan sedang, bervariasi tergantung faktor cuaca seperti angin dan kelembaban.  
    """)

    st.subheader("Kesimpulan")
    st.write("""
    Clustering ini membantu mengidentifikasi pola penyewaan sepeda berdasarkan cuaca dan jenis pengguna sepeda.
    """)

# Fungsi: Pola Penyewaan Berdasarkan Musim
def show_seasonal_analysis(filtered_days_df):
    if filtered_days_df.empty:
        st.warning("Data tidak tersedia untuk analisis musim.")
        return

    st.header("Pola Penyewaan Sepeda Berdasarkan Musim")
    season_rentals = filtered_days_df.groupby("musim")["total_penyewaan_sepeda"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="musim", y="total_penyewaan_sepeda", data=season_rentals, palette="coolwarm", ax=ax)

    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    st.pyplot(fig)

# Fungsi: Pola Penyewaan Berdasarkan Cuaca
def show_weather_analysis(filtered_days_df):
    if filtered_days_df.empty:
        st.warning("Data tidak tersedia untuk analisis cuaca.")
        return

    st.header("Pola Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
    weather_rentals = filtered_days_df.groupby("cuaca")["total_penyewaan_sepeda"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cuaca", y="total_penyewaan_sepeda", data=weather_rentals, palette="viridis", ax=ax)

    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    st.pyplot(fig)

# Fungsi: Dinamika Penyewaan Sepeda
def show_rentals_trends(filtered_days_df):
    if filtered_days_df.empty:
        st.warning("Data tidak tersedia untuk tren penyewaan.")
        return

    st.header("Dinamika Penyewaan Sepeda: Pengguna Terdaftar vs Kasual")
    rentals_trends = filtered_days_df.groupby("tanggal")[["penyewa_terdaftar", "penyewa_kasual"]].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="tanggal", y="penyewa_terdaftar", data=rentals_trends, label="Pengguna Terdaftar", color="b", ax=ax)
    sns.lineplot(x="tanggal", y="penyewa_kasual", data=rentals_trends, label="Pengguna Kasual", color="r", ax=ax)

    ax.set_title("Tren Penyewaan Sepeda: Pengguna Terdaftar vs Kasual")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Penyewaan")

    st.pyplot(fig)

# MAIN
def main():
    days_df, hours_df = load_data()
    start_date, end_date = sidebar_filters(days_df)
    st.sidebar.markdown("© 2025 by JOSUA SIANTURI")

    if start_date and end_date:
        filtered_days_df = filter_by_date(days_df, start_date, end_date)
        filtered_hours_df = filter_by_date(hours_df, start_date, end_date)

        st.write(f"**Rentang Waktu Dipilih:** {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")

        # TAMPILKAN METRIK SELALU
        show_total_metrics(filtered_days_df)

        # TAB MENU DI ATAS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Jam Sibuk & Sepi",
            "Pola Musim",
            "Pola Cuaca",
            "Tren Kasual vs Terdaftar",
            "Clustering"
        ])

        with tab1:
            show_hourly_analysis(filtered_hours_df)
        with tab2:
            show_seasonal_analysis(filtered_days_df)
        with tab3:
            show_weather_analysis(filtered_days_df)
        with tab4:
            show_rentals_trends(filtered_days_df)
        with tab5:
            show_clustering(filtered_days_df)

    else:
        st.info("Silakan pilih rentang waktu terlebih dahulu.")
        
if __name__ == "__main__":
    main()        